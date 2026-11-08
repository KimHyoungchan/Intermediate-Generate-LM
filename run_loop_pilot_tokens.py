#!/usr/bin/env python3
"""Token-count logging and live observation without changing the pilot engine."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parent
HEADER = re.compile(r"^(before|after): episode (\d+)/(\d+), target tokens=(\d+)")
RESOLVED = re.compile(r"\bresolved=(\d+)(?:/\d+)?")


def rooted(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def checksum(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def token_counts(context_tokens, target_tokens, generated_tokens):
    if not 0 <= generated_tokens <= target_tokens:
        raise ValueError("Generated token count is outside the episode")
    return {"context_tokens": context_tokens, "generated_tokens": generated_tokens,
            "target_tokens": target_tokens, "total_tokens": context_tokens + generated_tokens,
            "total_tokens_at_completion": context_tokens + target_tokens}


def suffix(counts):
    return (f" total_tokens={counts['total_tokens']}/{counts['total_tokens_at_completion']}"
            f" context_tokens={counts['context_tokens']}"
            f" generated_tokens={counts['generated_tokens']}/{counts['target_tokens']}")


class Catalog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.infos = None
        self.order = None

    def load(self):
        if self.infos is not None:
            return
        metadata = json.loads((self.directory / "resolved_config.json").read_text())
        source = yaml.safe_load(rooted(metadata["config"]["source_experiment_config"]).read_text())
        data = rooted(source["output_dir"]) / "data/test.jsonl"
        if checksum(data) != metadata["selection"]["test_sha256"]:
            raise ValueError("Source test data changed; refusing to guess token counts")
        episodes = [json.loads(s) for s in data.read_text().splitlines() if s.strip()]
        self.infos = {e["episode_id"]: {"context_tokens": len(e["context"]),
                      "target_tokens": len(e["targets"])} for e in episodes}
        self.order = metadata["selection"]["episode_ids"]

    def episode(self, episode_id):
        self.load()
        return self.infos[episode_id]

    def at_index(self, index):
        self.load()
        episode_id = self.order[index]
        return episode_id, self.infos[episode_id]

    def record(self, condition, episode_id, counts, source, event=None):
        logger_hash = checksum(Path(__file__).resolve())
        metadata = {"logging_source_sha256": logger_hash,
                    "definition": "total_tokens = original context + accepted/emitted target tokens; loop retries do not add tokens",
                    "pilot_engine_modified": False}
        path = self.directory / "token_logging.json"
        if not path.exists():
            temporary = self.directory / f"token_logging.{os.getpid()}.tmp"
            temporary.write_text(json.dumps(metadata, indent=2) + "\n")
            temporary.replace(path)
        row = {**(event or {}), "condition": condition, "episode_id": episode_id,
               **counts, "display_source": source, "logging_source_sha256": logger_hash}
        with (self.directory / "token_progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")


class TokenOutput:
    """Annotate complete console lines; leave pilot computation untouched."""
    def __init__(self, stream, catalog):
        self.stream = stream
        self.catalog = catalog
        self.buffer = ""
        self.condition = None
        self.episode_id = None
        self.info = None
        self.warned = False

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def annotate(self, line):
        header = HEADER.match(line)
        if header:
            self.condition = header[1]
            self.episode_id, self.info = self.catalog.at_index(int(header[2]) - 1)
            if int(header[4]) != self.info["target_tokens"]:
                raise ValueError("Episode header/data token count mismatch")
            counts = token_counts(**self.info, generated_tokens=0)
            return line + suffix(counts)
        match = RESOLVED.search(line)
        if match and self.info is not None:
            counts = token_counts(**self.info, generated_tokens=int(match[1]))
            self.catalog.record(self.condition, self.episode_id, counts, "launch",
                                {"console_line": line})
            return line + suffix(counts)
        return line

    def write(self, text):
        self.buffer += text
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            try:
                line = self.annotate(line)
            except (OSError, ValueError, KeyError, IndexError) as error:
                # Optional display must never terminate a long-running pilot.
                if not self.warned:
                    print(f"Token-count display unavailable: {error}", file=sys.stderr, flush=True)
                    self.warned = True
            self.stream.write(line + "\n")
        return len(text)

    def flush(self):
        self.stream.flush()

    def finish(self):
        if self.buffer:
            self.stream.write(self.buffer)
            self.buffer = ""
        self.flush()


def last_complete_event(path):
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        start = max(0, size - 65536)
        stream.seek(start)
        data = stream.read()
    lines = data.split(b"\n")[:-1]
    if start > 0:
        lines = lines[1:]
    for line in reversed(lines):
        if line.strip():
            return json.loads(line)
    return None


def watch(directory, once=False, interval=1.0):
    directory = rooted(directory)
    catalog = Catalog(directory)
    seen = {}
    if not once:
        print(f"Watching {directory}; Ctrl+C stops only this monitor.", flush=True)
    try:
        while True:
            for condition in ("before", "after"):
                event = last_complete_event(directory / condition / "progress.jsonl")
                if event is None:
                    continue
                key = (event["episode_id"], event["visited_blocks"])
                if seen.get(condition) == key:
                    continue
                seen[condition] = key
                info = catalog.episode(event["episode_id"])
                counts = token_counts(**info, generated_tokens=event["resolved_targets"])
                print(f"{condition}: target={event['target_index']} loops={event['target_wraps']} "
                      f"total_loops={event['total_wraps']} blocks={event['visited_blocks']}"
                      + suffix(counts) + f" elapsed={event['elapsed_seconds']:.1f}s", flush=True)
                catalog.record(condition, event["episode_id"], counts, "watch", event)
            if once:
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        return


def option_value(args, name):
    for index, value in enumerate(args):
        if value == name:
            return args[index + 1] if index + 1 < len(args) else None
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def main():
    observer = argparse.ArgumentParser(add_help=False)
    observer.add_argument("--watch", type=Path)
    observer.add_argument("--once", action="store_true")
    observer.add_argument("--interval", type=float, default=1.0)
    args, forwarded = observer.parse_known_args()
    if args.watch:
        if forwarded:
            observer.error("Watch mode accepts only --watch, --once and --interval")
        if args.interval <= 0:
            observer.error("--interval must be positive")
        watch(args.watch, once=args.once, interval=args.interval)
        return
    if args.once or args.interval != 1.0:
        observer.error("--once and --interval require --watch")
    config_path = option_value(forwarded, "--config")
    sys.path.insert(0, str(ROOT / "src"))
    from loop_pilot.cli import main as pilot_main
    sys.argv = [sys.argv[0], *forwarded]
    if not config_path or "--help" in forwarded or "-h" in forwarded:
        print("Token display enabled. Live monitor: --watch RUN_DIRECTORY [--once] [--interval 1]")
        pilot_main()
        return
    config = yaml.safe_load(Path(config_path).resolve().read_text())
    directory = rooted(option_value(forwarded, "--output-dir") or config["output_dir"])
    original = sys.stdout
    output = TokenOutput(original, Catalog(directory))
    sys.stdout = output
    try:
        pilot_main()
    finally:
        output.finish()
        sys.stdout = original


if __name__ == "__main__":
    main()
