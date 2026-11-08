from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_loop_pilot_tokens import TokenOutput, token_counts, last_complete_event


class FakeCatalog:
    def __init__(self):
        self.saved = []

    def at_index(self, index):
        return "episode", {"context_tokens": 32, "target_tokens": 8}

    def record(self, *values):
        self.saved.append(values)


class TokenLoggingTests(unittest.TestCase):
    def test_prompt_and_generated_tokens_are_counted_separately(self):
        self.assertEqual(token_counts(32, 8, 1), {
            "context_tokens": 32, "generated_tokens": 1, "target_tokens": 8,
            "total_tokens": 33, "total_tokens_at_completion": 40})

    def test_loop_retries_do_not_inflate_total_tokens_and_hit_increases_it(self):
        stream = StringIO()
        output = TokenOutput(stream, FakeCatalog())
        output.write("after: episode 1/1, target tokens=8; per-token cap=NONE\n")
        output.write("  target=1 loops=32 resolved=1 blocks=770 elapsed=1.2s\n")
        output.write("  target=1 loops=64 resolved=1 blocks=1538 elapsed=2.3s\n")
        output.write("  target=1 loops=65 resolved=2 blocks=1562 elapsed=2.4s\n")
        lines = stream.getvalue().splitlines()
        self.assertIn("total_tokens=32/40", lines[0])
        self.assertIn("total_tokens=33/40", lines[1])
        self.assertIn("total_tokens=33/40", lines[2])
        self.assertIn("total_tokens=34/40", lines[3])

    def test_split_console_writes_and_final_token_count(self):
        stream = StringIO()
        output = TokenOutput(stream, FakeCatalog())
        output.write("after: episode 1/1, target tokens=8; per-token cap=NONE\n")
        output.write("  complete: resolved=")
        output.write("8/8 observed_loops=10\n")
        self.assertIn("total_tokens=40/40", stream.getvalue())

    def test_unfinished_json_write_is_not_treated_as_an_event(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            path.write_text(json.dumps({"resolved_targets": 1}) + '\n{"resolved_targets":')
            self.assertEqual(last_complete_event(path), {"resolved_targets": 1})


if __name__ == "__main__":
    unittest.main()
