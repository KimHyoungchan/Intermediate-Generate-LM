from __future__ import annotations

import csv
import json
from pathlib import Path
import yaml

from layerwise_lm.common import file_hash, read_json, write_json


def training_description(source):
    if source.get("iteration_policy") == "global":
        return ("Checkpoint는 next-token 정답과 시퀀스 전체 누적 iteration <= 현재 target index+1 규칙으로 학습했다. "
                "학습에서는 예산 소진 후 정답 입력을 공급했지만, 이 pilot은 강제 공급 없이 직접 맞혀야 다음 token으로 진행한다.")
    budget = source.get("max_visits_per_token")
    if budget is not None:
        return f"Checkpoint는 token당 {budget}회 방문 예산으로 학습했으며, 이 pilot에서는 그 token별 제한을 해제했다."
    return "Checkpoint의 학습 규칙은 source_experiment_config를 참조한다. 이 pilot에서는 재학습하지 않는다."


def report(output, config):
    from .runner import identity, csv_write
    conditions = {}
    episode_rows = []
    for name in ("before", "after"):
        folder = output / name
        if not (folder / "summary.json").exists():
            continue
        summary = read_json(folder / "summary.json")
        if any(summary.get(k) != v for k, v in identity(config).items()):
            raise ValueError("Report config/source differs from saved loop pilot")
        for file in ("episodes", "tokens"):
            if file_hash(folder / f"{file}.csv") != summary[f"{file}_sha256"]:
                raise ValueError(f"Loop pilot artifact changed: {name}/{file}")
        conditions[name] = summary
        with (folder / "episodes.csv").open() as stream:
            for row in csv.DictReader(stream):
                episode_rows.append({"condition": name, **row})
    if not conditions:
        raise FileNotFoundError("No loop pilot results yet")
    if len(conditions) == 2:
        assert conditions["before"]["selection"] == conditions["after"]["selection"]
    source_path = Path(next(iter(conditions.values()))["source_experiment_config"])
    source_config = yaml.safe_load(source_path.read_text())
    csv_write(output / "episode_comparison.csv", episode_rows)
    write_json(output / "summary.json", {**identity(config), "conditions": conditions})
    def fmt(value):
        return "미확인" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)
    lines = ["# 완주까지 필요한 루프 수 pilot", "",
        "Token별 방문 한도 없음. 기존 상태와 cache를 유지하며 실제 다음 token을 맞히면 같은 layer에서 이어간다.", "",
        "루프 1회 = 같은 target에서 실제로 마지막 layer → 첫 layer 전환이 관측된 것.", "",
        "| 조건 | 완주/선택 시퀀스 | 완주 시퀀스 평균 루프 | 처리 target | 한 token의 최대 관측 루프 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for name, s in conditions.items():
        lines.append(f"| {name} | {s['completed_episodes']}/{s['selected_episodes']} | "
                     f"{fmt(s['mean_wraps_to_completion_completed_only'])} | {s['resolved_targets']}/{s['requested_targets_selected']} | "
                     f"{s['max_observed_wraps_for_one_token']} |")
    lines += ["", "| 조건 | Episode | 결과 | 맞힌/요청 token | 완주 루프 또는 관측 하한 | 중단 이유 |",
              "| --- | --- | --- | ---: | ---: | --- |"]
    for row in episode_rows:
        complete = row["outcome"] == "complete"
        bound = row["observed_wraps"] if complete else ">= " + row["observed_wraps"]
        lines.append(f"| {row['condition']} | {row['episode_id'][:10]} | {'완주' if complete else '미완주·검열'} | "
                     f"{row['resolved_targets']}/{row['requested_targets']} | {bound} | {row['stop_reason']} |")
    lines += ["", "미완주 행의 숫자는 관측한 루프 수의 하한이며, 그 횟수면 완주한다는 뜻이 아니다. 완주한 표본이 없으면 필요한 평균 루프 수는 미확인이다.", "",
        "시간 제한은 episode 전체에 대한 외부 watchdog이다. 96회 token 예산이나 token별 강제 종료는 사용하지 않는다. 0초 설정은 시간 제한을 해제한다.", "",
        "평가는 데이터 정답으로 분기를 고르는 oracle 방식이다. 학습 전/후는 같은 test episode를 사용한다. " + training_description(source_config), "",
        "동일한 수치 상태가 재관측됐는지도 저장한다. 이는 관측 지표이며 자동 종료나 완주 불가능 판정에 사용하지 않는다.", "",
        "![Sequence progress](progress.png)", ""]
    (output / "report.md").write_text("\n".join(lines))
    plot(output, conditions)
    print(f"Loop completion report: {output / 'report.md'}", flush=True)


def plot(output, conditions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(conditions), 1, figsize=(9, 4 * len(conditions)), squeeze=False)
    for ax, name in zip(axes[:, 0], conditions):
        paths = {}
        with (output / name / "progress.jsonl").open() as stream:
            for line in stream:
                event = json.loads(line)
                paths.setdefault(event["episode_id"], [(0, 0)]).append((event["total_wraps"], event["resolved_targets"]))
        with (output / name / "episodes.csv").open() as stream:
            episodes = list(csv.DictReader(stream))
        for episode in episodes:
            points = paths.get(episode["episode_id"], [(0, 0)])
            points.append((int(episode["observed_wraps"]), int(episode["resolved_targets"])))
            x, y = zip(*points)
            line, = ax.step(x, y, where="post", label=episode["episode_id"][:8])
            ax.plot(x[-1], y[-1], marker="o" if episode["outcome"] == "complete" else "x", color=line.get_color())
        requested = max(int(e["requested_targets"]) for e in episodes)
        ax.axhline(requested, color="gray", linestyle="--", label="all targets complete")
        ax.set_ylim(-0.2, requested + 0.4)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(left=0)
        ax.set_xlabel("Observed L-to-1 loops (symlog scale)")
        ax.set_ylabel("Resolved target tokens")
        ax.set_title(f"{name}: circle = complete, cross = censored")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"progress.{suffix}", dpi=160)
    plt.close(fig)
