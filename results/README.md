# 공개 결과 snapshot

원래 실험의 `runs/<name>/`에 대응하는 읽기 전용 결과를 `results/<name>/`에 보존했다. 학습 데이터·weight·checkpoint·전체 block trace는 제외했다. 새 실행 결과는 `runs/`에 생성된다.

## 주요 결과

| 경로 | 내용 | 해석 범위 |
| --- | --- | --- |
| [pilot_05b](pilot_05b/comparison.md) | 0.5B 표준 경로의 중간 readout 분석 | Frozen backbone의 출력 해석 실험 |
| [main_7b](main_7b/comparison.md) | 7B readout 분석, train 100만 targets | 실제 주입·순환의 성능과 구분 |
| [routed_05b](routed_05b/comparison.md) | 초기 실제 주입·순환 | 고정 방문 예산, 실패 시 episode 종료 |
| [routed_05b_v96](routed_05b_v96/comparison.md), [routed_7b_v96](routed_7b_v96/comparison.md) | Token당 96회 방문 예산 | Unreached target과 실패 비용 포함 |
| [training_revision_05b](training_revision_05b/comparison.md), [training_revision_7b](training_revision_7b/comparison.md) | 누적 iteration index+1, 1,024 train targets | 실패 뒤 정답 입력으로 다음 target 진행 |
| [training_parallel_05b](training_parallel_05b/comparison.md) | Train 1,048,576 targets × 3 epochs | 완료된 0.5B 확대 실험 |
| [training_parallel_7b](training_parallel_7b/data/manifest.json) | 확대 표본 준비 | 학습 완료 결과 없음 |
| [loop_pilot_05b_wait](loop_pilot_05b_wait/report.md) | 장시간 loop 관측 | 시간·중단에 의한 검열 포함 |
| [0.5B strict loop](training_revision_05b/strict_loop_check/report.md), [7B strict loop](training_revision_7b/strict_loop_check/report.md) | 새 소규모 checkpoint의 강제 공급 없는 관측 | 각각 2 episodes, 짧은 관측 시간 |

`smoke_*`, `routed_smoke_*`, `loop_pilot_smoke*`, `training_parallel_smoke*`는 구현 동작 확인 기록이다. 모든 snapshot의 완료 상태와 종류는 [index.json](index.json)에 있다. 선택하지 않은 label 정렬의 임시 점검은 방법 문서에서 참조하는 짧은 보고서만 보존했다. 미완료 초기 검사와 token logging 전용 임시 실행은 제외했다.

## 포함한 결과

- 학습 전후 summary, comparison, 주요 layer/target/episode CSV.
- 데이터 표본의 manifest와 SHA256. 원문 또는 tokenized train JSONL은 제외.
- 학습 완료량, checkpoint hash, 환경·GPU 정보와 validation 경과 log.
- 실행 설정과 핵심 도표. 학습 checkpoint 자체는 제외.

Snapshot을 재실행 출력 폴더로 복사하여 `--resume`하는 방식은 지원하지 않는다. 재현은 저장소 루트의 명령으로 새로운 `runs/`에서 시작한다. 각 실험의 source/config hash를 비교할 수 있도록 실행 코드와 설정 파일의 bytes는 원본 그대로 보존했다. 문서의 링크만 공개 snapshot 위치에 맞췄다.
