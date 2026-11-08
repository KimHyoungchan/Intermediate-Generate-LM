# 완주까지 필요한 루프 수를 관측하는 pilot

2026-11-02. 목적은 **동일한 시퀀스와 상태를 계속 이어갈 때 몇 번 L→1 재순환해야 완주하는지** 측정하는 것이다. Token별 96회 제한을 늘려 재실행하는 실험과 구분한다.

## 동작

- 기존 `run_routed.py`의 같은-layer token 주입·state-carrying wrap 경로를 그대로 사용한다.
- 평가 중 token별 block 방문 한도를 **해제**한다. 성공한 token만 즉시 입력에 추가하고 같은 layer에서 다음 예측을 시작한다.
- 같은 token을 계속 틀리면 hidden state와 KV cache를 유지하며 반복한다. Loop threshold마다 상태를 reset하거나 시퀀스를 재시작하지 않는다.
- **루프 1회는 같은 target에서 마지막 layer를 나온 후 첫 layer 실행이 실제 관측된 횟수**다. 단순한 `block 방문 수 / layer 수`로 계산하지 않는다.
- 모든 target token을 맞히면 해당 시퀀스를 즉시 완료 처리한다.

이 pilot은 **기존 96회 예산으로 학습한 checkpoint를 재학습 없이 사용**한다. 현재 설정의 학습 전 초기 모델과 학습 후 모델을 같은 test episode에서 비교한다. 정답 판정은 데이터의 실제 다음 token을 사용하는 oracle 방식이다.

## 무한 대기와 미완주 처리

Token별 횟수 제한은 없다. 기본 pilot은 episode당 60초의 외부 시간 watchdog을 두며, 이때까지 끝나지 않으면 **시간 내 미완주(right-censored)**로 저장한다. 96회 실패로 끝내거나 정답을 강제로 넣지 않는다.

- 완주한 경우: 정확한 완주 루프 수 기록.
- 시간 내 미완주: `관측 루프 >= N, 필요 루프 수 미확인`으로 기록. N회면 완주한다는 뜻이 아니다.
- 아직 시작하지 못한 다음 target: `unreached`.
- 완주 시퀀스가 0개면 평균 필요 루프 수도 미확인이다. 미완주 표본을 0이나 timeout 횟수로 대체해 평균 내지 않는다.

같은 수치 상태가 다시 나타나는지도 관측하지만, 이를 자동 종료 기준이나 완주 불가능 판정에 사용하지 않는다. 관측 상태 hash는 현재 token의 최근 4,096개 distinct wrap-boundary 상태만 유지한다. 관측을 끄려면 YAML의 `observe_repeated_states`를 false로 둔다.

## 환경

```bash
source /home/hyoungchan38/anaconda3/etc/profile.d/conda.sh
conda activate ppo_gpu
cd /home/hyoungchan38/layerwise-lm-experiments
```

GPU는 물리 번호 1·2·3만 허용하며, 새 launcher도 CUDA 초기화 전에 UUID로 격리한다.

## 기본 pilot: 0.5B

```bash
python run_loop_pilot_tokens.py --config configs/loop_pilot_05b.yaml --gpu 1
```

현재 저장된 공식 test episode 중 첫 4개, 각각 context 32 / continuation 8 tokens를 사용한다. 학습 전과 학습 후를 모두 실행한다. Episode당 최대 관측 시간은 60초라 두 조건 합계 최대 약 8분이며, 모델 로딩·보고서 시간은 별도다. 먼저 완주하면 즉시 다음 episode로 넘어간다.

## 7B

```bash
python run_loop_pilot_tokens.py --config configs/loop_pilot_7b.yaml --gpu 3
```

같은 규칙으로 기존 `runs/routed_7b_v96/training/best.pt`를 사용한다. 이 명령은 재학습을 수행하지 않는다.

## 더 오래 관측하기

Episode당 5분으로 늘리고 결과를 별도로 저장한다.

```bash
python run_loop_pilot_tokens.py --config configs/loop_pilot_05b.yaml --gpu 1 \
  --max-seconds 300 --output-dir runs/loop_pilot_05b_300s
```

완주할 때까지 **시간 제한도 없이**, 학습 후 모델의 시퀀스 하나를 계속 실행하려면:

```bash
python run_loop_pilot_tokens.py --config configs/loop_pilot_05b.yaml --gpu 1 \
  --condition after --episodes 1 --max-seconds 0 \
  --output-dir runs/loop_pilot_05b_wait
```

이 모드에는 자동 token/loop/시간 한도가 없다. 완주가 보장되는 모델은 아니므로 끝나지 않을 수 있다. `Ctrl+C`로 중단하면 가능한 경우 현재 관측 수를 미완주로 저장하고 보고서를 만든다. 계속 기다린다는 사실 자체를 수렴이나 완주 보장으로 해석하지 않는다.

다른 시간·episode 수를 사용하려면 별도 `--output-dir`을 지정한다. `--resume`은 동일 설정의 완료 조건 결과를 재사용한다. 중단된 hidden/cache를 디스크에서 이어가는 기능은 아니며, 미완료 조건은 다시 시작한다.

학습 전만 실행하려면 `--condition before`, 학습 후만은 `--condition after`를 붙인다. 저장된 결과로 보고서만 다시 만들 때는 동일한 설정과 override로 `--condition report`를 사용한다.

## 출력

진행 로그에 총 token 수를 함께 표시한다.

```text
target=1 loops=64 resolved=1 blocks=1538 ... total_tokens=33/40 context_tokens=32 generated_tokens=1/8
```

- `total_tokens`: 최초 context + 지금까지 정답을 맞혀 생성한 target token 수.
- `/40`: 이 예시의 context 32 + 요청한 target 8개를 모두 마쳤을 때의 총 token 수.
- `generated_tokens`: 생성한 target / 요청한 target. 같은 token을 재시도하는 루프는 token 수를 증가시키지 않는다.

현재 실행을 중단하지 않고 옆 터미널에서 같은 정보를 보려면:

```bash
python run_loop_pilot_tokens.py --watch runs/loop_pilot_05b_wait
```

`--once`는 한 번만 현재 값을 출력한다. `--interval 0.5`로 표시 간격을 바꿀 수 있다. 모니터의 `Ctrl+C`는 모니터만 종료하며 모델 실행은 유지한다. 새 실행은 위의 `run_loop_pilot_tokens.py` 명령을 사용하면 원래 진행 로그에 token 수가 붙는다. 기존 `run_loop_pilot.py`도 계산 경로·기존 결과 재현용으로 유지한다.

로그 실행기는 원래 pilot의 모델 계산과 source 파일을 수정하지 않는다. 기존 실행의 source hash도 보존한다. 표시한 값은 결과 폴더의 `token_progress.jsonl`, 표시 기능의 정의·source hash는 `token_logging.json`에 별도로 저장한다.

기본 0.5B는 `runs/loop_pilot_05b/`, 7B는 `runs/loop_pilot_7b/`에 저장한다.

```text
report.md                  # 먼저 볼 완주·미완주 및 루프 수 표
summary.json
episode_comparison.csv
resolved_config.json
progress.png / progress.pdf
before/ 또는 after/
  summary.json
  episodes.csv             # 시퀀스별 완주 루프 또는 관측 하한
  tokens.csv               # token별 실제 방문·루프·hit/censored/unreached
  progress.jsonl           # 매 실제 L→1 전환 및 정답 시점의 진행 기록
```

`wraps_to_completion`은 완주한 시퀀스에서만 숫자다. 미완주는 null이며 `observed_wraps`가 관측 하한이다. Token의 `status=hit`인 경우 `wraps`는 정답까지의 정확한 재순환 수이고, `censored`는 지금까지의 관측치다.

## 검증

```bash
python -m unittest discover -s tests_loop_pilot -v
```

5개 CPU 검사에서 다음을 확인했다.

- 96회를 넘겨 상태 reset 없이 계속 실행하고, 100회 재순환 후 완주하는 scripted 경로.
- 실제 관측되지 않은 예정 wrap을 루프에 포함하지 않음.
- 시간 제한의 미완주를 완료나 실패로 오인하지 않음.
- 미완주 표본을 완주 평균에 포함하지 않음.

실제 0.5B에서는 각 조건 2 episodes × 최대 5초의 최소 검증을 실행했다. 두 조건 모두 96 block 방문을 훨씬 넘어 100회 이상의 재순환까지 같은 경로를 이어갔고, 시간 내 완주한 시퀀스는 없었다. 따라서 이 최소 검증에서 필요한 완주 루프 수는 **미확인**이다.

- [최소 실제 실행 보고서](results/loop_pilot_smoke/report.md)
- [7B 최소 실제 실행 보고서](results/loop_pilot_smoke_7b/report.md): 물리 GPU 3, 각 조건 1 episode를 2초 관측. 학습 전 22회·학습 후 27회 재순환까지 같은 상태를 이어갔고, 완주 루프 수는 미확인이었다.
- [기존 실제 경로 학습 설정](ROUTED_EXPERIMENT.md)

`--resume`으로 완료된 최소 실행의 결과 재사용과 보고서 재생성도 확인했다. 중간 cache/state의 재개를 뜻하지 않는다.

새 누적 iteration 학습 모델을 위해 보고서가 source config의 학습 규칙을 읽도록 개선했다. 이전 pilot 결과는 보존한다. 소스 hash가 다른 과거 실행을 재사용하려고 하면 검사가 중단하므로, 새 모델의 비교에는 `configs/loop_pilot_revision_05b.yaml`, `configs/loop_pilot_revision_7b.yaml`의 별도 결과 폴더를 사용한다.

기존 `routed_lm/`, `src/layerwise_lm/`, checkpoint를 수정하지 않고 별도 `loop_pilot/`에 구현했다. 기존 학습 및 source hash 검증을 보존한다.
