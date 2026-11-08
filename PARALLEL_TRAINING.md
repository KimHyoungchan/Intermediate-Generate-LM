# GPU 여러 장을 사용하는 순환 경로 학습

물리 GPU **1·2·3만** 사용하는 동기식 data parallel 실행기: `run_training_parallel.py`. 0.5B와 7B 모두 지원한다. 기존 단일 GPU 실행기와 완료 checkpoint의 소스 해시를 보존하기 위해 `parallel_training/`에 별도 구현했다.

## 실행

```bash
source /home/hyoungchan38/anaconda3/etc/profile.d/conda.sh
conda activate ppo_gpu
cd /home/hyoungchan38/layerwise-lm-experiments

# 0.5B 한 모델을 GPU 1·2·3에 분산해 학습 전후 비교
python run_training_parallel.py --config configs/training_parallel_05b.yaml --gpus 1,2,3

# 7B 한 모델을 GPU 1·2·3에 분산해 학습 전후 비교
python run_training_parallel.py --config configs/training_parallel_7b.yaml --gpus 1,2,3
```

각 명령은 prepare → 학습 전 평가 → 분산 학습 → 학습 후 평가 → 비교 보고서를 순서대로 수행한다. 위 두 명령은 같은 GPU들을 사용하므로 순서대로 실행한다. 학습 단계가 여러 GPU를 사용하며, validation과 학습 전후 test 평가는 선택한 첫 번째 GPU에서 수행한다. Prepare/compare는 CPU를 사용한다.

GPU 두 장이나 한 장도 같은 실행기로 선택할 수 있다. 실행 구성이 결과 폴더에 고정되므로 다른 GPU 수·학습량을 비교할 때에는 새 output-dir을 사용한다.

```bash
python run_training_parallel.py --config configs/training_parallel_05b.yaml --gpus 1,3 --output-dir runs/training_parallel_05b_two
python run_training_parallel.py --config configs/training_parallel_05b.yaml --gpus 2 --output-dir runs/training_parallel_05b_one
```

## 확대 학습 설정

| 항목 | 0.5B / 7B 공통 |
| --- | ---: |
| Train episodes | 131,072 |
| 서로 다른 train 정답 위치 | 1,048,576 tokens |
| Validation | 512 episodes / 4,096 targets |
| Test | 1,024 episodes / 8,192 targets |
| Epochs | 3 |
| 3 epochs 누적 train 정답 처리 예정 수 | 3,145,728 |
| Episode 길이 | Context 32 + targets 8 |
| LoRA rank / learning rate | 8 / 0.0001 |
| Gradient accumulation | 전체 GPU 합산 최소 8,192 head 방문 |
| Validation 간격 | 256 optimizer updates 및 마지막 update |
| BPTT | Episode 전체 graph 유지 |

이는 전체 WikiText-2의 train 2,517,232 tokens 중에서 **1,048,576개의 중복 없는 target 위치를 추출하는 설정**이다. 전체 train coverage와 다르다. Epoch마다 같은 표본의 순서를 바꾸며, 3 epoch를 서로 다른 정답 314만 개로 해석하지 않는다. 전체 split과 이전 1,024-target 실행의 차이는 [DATASET_SCALE.md](DATASET_SCALE.md)에 있다.

2026-11-03 확인: 0.5B 확대 학습과 학습 전후 test 평가를 완료했다. 7B 확대 폴더에는 데이터 준비 기록만 있으며 학습 완료 기록은 없다. Train target 위치의 중복을 금지하고, 문서 균형 순환 선택·random offset·seed 42 규칙을 사용한다.

필요하면 YAML 또는 CLI에서 명시적으로 학습량을 변경한다.

```bash
python run_training_parallel.py --config configs/training_parallel_05b.yaml --gpus 1,2,3 --epochs 5 --train-episodes 131072 --accumulation-visits 8192 --output-dir runs/training_parallel_05b_e5
```

`--train-episodes N`의 정답 수는 N × continuation_tokens다. 데이터에서 중복 없이 추출할 수 있는 수보다 큰 값을 요청하면 실패한다. Config 변경은 새로운 output-dir이 필요하다.

## 병렬 처리와 학습 규칙

각 worker에 전체 모델을 한 개씩 로드하고, 서로 다른 episode를 배정한다. 전역 순서를 한 번 shuffle한 뒤 각 round에서 rank별로 episode 하나를 처리한다. 마지막 round의 episode 수가 GPU 수보다 적으면 남는 rank는 episode 계산을 쉬고 통신에는 참여한다. Episode 복제나 마지막 표본 버리기를 하지 않는다.

각 episode 내부에서는 기존 규칙을 유지한다. 입력 A의 정답은 B인 next-token CE다. 정답이면 같은 layer에 다음 token을 주입한다. 실패하면 다음 layer, 마지막 layer에서는 전체 누적 iteration과 target index+1을 비교한다. 제한에 도달하면 loss를 남기고 정답을 입력으로 주어 layer 1에서 다음 token 학습을 계속한다. Episode 내부 hidden/KV graph를 유지하므로 뒤쪽 token loss가 앞쪽 상태에도 역전파된다.

GPU마다 head 방문 수가 달라질 수 있으므로 gradient는 다음 규칙으로 동기화한다.

`전체 gradient = SUM(각 GPU의 CE 합 gradient) / SUM(각 GPU의 실제 head 방문 수)`

GPU별 CE 평균을 다시 평균하지 않는다. 동적 경로에서 일부 rank가 사용하지 않은 parameter도 통신에 참여한다. 모든 rank가 사용하지 않은 parameter는 grad=None을 유지해 AdamW momentum의 불필요한 적용을 방지한다. Gradient 합산 뒤 전체 gradient를 clip하고 모든 rank에서 같은 optimizer step을 수행한다.

Global accumulation 기준은 episode round를 마친 뒤 검사한다. 따라서 지정된 방문 수보다 조금 더 모은 뒤 update할 수 있으며, 한 episode 중간에는 optimizer를 바꾸지 않는다. GPU 수가 달라지면 episode 묶음과 update 시점이 달라질 수 있으므로 단일 GPU와 학습 궤적이 완전히 같다고 가정하지 않는다.

통신은 PyTorch distributed/NCCL의 명시적 all-reduce를 사용한다. 각 프로세스는 CUDA 초기화 전에 지정한 물리 GPU의 UUID 하나만 보도록 제한한다. 코드 내부의 cuda:0은 그 프로세스에 배정된 허용 GPU다. `--gpus 0`, 중복 GPU, 1·2·3 이외 번호는 실행 전에 거부한다. 각 GPU에 모델 전체를 로드하므로 GPU별 모델 메모리는 필요하다.

## 결과와 재사용

- `runs/training_parallel_05b/comparison.md`: 학습 전후 결과.
- `runs/training_parallel_05b/training/done.json`: 전체 처리 수, update 수, 물리 GPU UUID, 각 rank의 처리 수·최대 메모리·동일 parameter hash.
- `training/coverage.rankN.jsonl`: epoch별 episode 배정과 처리 수.
- `training/log.jsonl`: rank 0의 validation 결과.
- `training/best.pt`: validation 정답률 우선, CE 차선으로 선택한 checkpoint.

7B 경로는 `training_parallel_7b`다. 큰 설정에서는 모든 head 방문 trace의 대량 기록을 끄고 episode coverage는 기록한다. 작은 smoke는 `trace_training: true`다.

완료된 단계는 다음과 같이 검증 후 재사용한다.

```bash
python run_training_parallel.py --config configs/training_parallel_05b.yaml --gpus 1,2,3 --resume
```

현재 resume는 완료된 단계 재사용이며, 중단된 학습의 optimizer step 복원은 지원하지 않는다. 중단된 train은 해당 설정의 초기 상태부터 다시 학습한다. 같은 설정으로 CPU prepare만 완료한 폴더에서는 위 전체 명령을 실행하면 된다.

새 checkpoint는 기존 loop pilot에서도 사용할 수 있다. Loop pilot YAML의 `source_experiment_config`를 `runs/training_parallel_05b/resolved_config.yaml` 또는 7B 경로로 지정하고 새로운 loop output-dir을 사용한다. 학습 전후 test 평가는 실패 뒤 정답을 공급하며, 강제 공급 없는 loop 완주 평가는 별도로 수행한다.

## 검증

```bash
python -m unittest discover -s tests_parallel_training -v
python -m unittest discover -s tests_training_revision -v
python run_training_parallel.py --config configs/training_parallel_smoke.yaml --gpus 1,2,3
python run_training_parallel.py --config configs/training_parallel_smoke_7b.yaml --gpus 1,2,3
```

CPU 검사에서는 불균등한 visit 수, rank별 미사용 parameter, 마지막 idle rank를 포함한 실제 2-process Gloo gradient 동기화 결과를 중앙집중식 AdamW 두 step과 비교한다. Episode 배정의 중복·누락 방지와 금지 GPU도 검사한다. 기존 next-token/iteration/gradient 경로 검사 7개를 함께 통과했다.

실제 GPU smoke는 7 episodes × 3 targets × 2 epochs = 42 targets로 통신·gradient·마지막 불균등 round·저장·복원을 확인한다. 이 42-target 검증은 확대 학습의 성능 결과가 아니다. 실제 실행 결과는 각 smoke 폴더의 `training/done.json`을 기준으로 한다.

2026-11-02 실제 검증 완료:

| 검증 | 0.5B | 7B |
| --- | --- | --- |
| 물리 GPU | 1, 2, 3 | 1, 2, 3 |
| 처리 targets / updates | 42/42 / 3 | 42/42 / 3 |
| Epoch·episode 중복 및 누락 | 없음, 14개 조합 | 없음, 14개 조합 |
| 세 rank의 최종 parameter hash | 동일 | 동일 |
| 원본 weight | 불변 | 불변 |
| Checkpoint 복원·학습 후 평가·비교 | 완료 | 완료 |

- [0.5B 실행 근거](results/training_parallel_smoke/training/done.json)
- [7B 실행 근거](results/training_parallel_smoke_7b/training/done.json)

분산 관련 CPU 검사 3개와 기존 경로 검사 7개가 통과했다. 기존 단일 GPU checkpoint의 identity도 그대로 유효하다. 두 확대 설정의 prepare를 완료했으며, train/validation/test JSONL의 SHA256이 모델 사이에서 각각 일치한다. 따라서 준비된 확대 표본은 두 모델에서 동일하다. 확대 실행의 최신 완료 상태는 아래 결과를 기준으로 한다.

## 0.5B 확대 실행 결과 — 2026-11-03 확인

물리 GPU 1·2·3에서 393,216 episodes와 누적 3,145,728 targets 전체를 처리했다. 3 epochs, 8,637 updates, 기록된 학습 시간 약 25.75시간이며 validation으로 선택한 checkpoint는 update 8,192다. 원본 weight는 고정했고 q/v LoRA, entry/memory, wrap, exit 보정 파라미터를 학습했다. Transformer 본체 전체를 미세조정한 실행은 아니다.

같은 공식 test 1,024 episodes·8,192 targets에서 다음을 관측했다.

| 지표 | 학습 전 | 학습 후 |
| --- | ---: | ---: |
| 직접 정답 target | 651 (7.95%) | 2,537 (30.97%) |
| 강제 정답 공급 target | 7,541 | 5,655 |
| 직접 정답 target의 평균 실제 block 방문 | 21.31 | 5.62 |
| 실패 비용을 포함한 전체 target 평균 block 방문 | 24.07 | 22.95 |
| 강제 공급 없이 8 targets 모두 정답인 episode | 0/1,024 | 2/1,024 |
| 첫 target 직접 정답 | 435/1,024 (42.48%) | 284/1,024 (27.73%) |
| 같은 target을 계속 탐색한 실제 wrap action 수 | 629 | 2,020 |

전체 직접 정답 비율은 +23.02%p, 전체 block 방문은 4.65% 감소했다. 양쪽 모두 hit인 동일한 327개 target에서도 실제 평균 방문은 21.32→4.69였다. 다만 이후 token의 경로는 학습 전후 달라질 수 있다.

평가는 oracle로 정답을 판정하고, iteration 제한 후에는 해당 정답을 입력으로 공급하여 다음 target 평가를 계속한다. 따라서 30.97%는 정답을 모르는 자유 생성의 정확도가 아니다. 8,192개를 모두 처리한 것은 평가 진행 완료를 의미한다. 학습 후에도 69.03%의 target은 정답을 공급했으며, 첫 target 성공 비율은 하락했다. 이번 checkpoint의 강제 공급 없는 별도 unlimited-loop 평가 결과는 확인되지 않았다.

Wrap 수는 증가했으므로 학습 후 순환이 줄었다고 해석하지 않는다. Trace를 기록한 평가 시간은 약 307.84→300.85초였으나, 표준 AR와 같은 품질·작업량을 맞춘 실제 가속 benchmark는 아니다. 기존 표준 forward의 최초 정답 깊이 관측과도 데이터·경로·지표가 다르다.

Config/source/checkpoint 및 test 표본 identity와 trace의 방문 수·정답 수를 대조했다. 세 rank의 최종 parameter hash도 동일했다. [실행 결과표](results/training_parallel_05b/comparison.md), [학습 완료 기록](results/training_parallel_05b/training/done.json), [test 상세](results/training_parallel_05b/after_train/tokens.csv).
