# 실패 이후에도 다음 token을 학습하는 새 iteration 경로

**확대·분산 학습 명령:** 기존 소규모 결과를 보존하며 새 실행기 `run_training_parallel.py`가 물리 GPU 1·2·3에서 동기식 data parallel 학습을 수행한다. Train 1,048,576 targets × 3 epochs의 새 구성과 검증은 [PARALLEL_TRAINING.md](PARALLEL_TRAINING.md)를 따른다. 아래 단일 GPU 완료 결과는 이전 소규모 실행이다.

**학습량 정정:** 아래 완료 결과는 전체 WikiText-2 학습이 아니라 train 1,024 targets만 사용한 동작 검증이다. 전체 train은 실측 2,517,232 Qwen tokens로, 사용한 정답 수는 그 약 0.0407%다. `1 epoch`는 추출한 작은 표본 기준이다. [전체 split 규모·실제 사용량](DATASET_SCALE.md).

2026-11-02. 사용자 요청에 따라 기존의 '예산 소진 시 시퀀스 중단'을 학습에서 변경하는 구현이다. **사용자 확정: 입력 A의 정답은 B(`next_token`), iteration은 시퀀스 전체 누적(`global`).** 아래 두 기술 검증은 확인 전 서로 다른 해석의 코드 경로를 검사한 최소 실행이며, 본 실험의 설정은 next_token + global이다.

**상태: 확정 규칙으로 0.5B·7B 학습, 학습 전후 평가, 별도의 정답 강제 공급 없는 짧은 loop 검증까지 완료.** 기존 사용량 제한으로 로그 확인이 중단된 동안에도 이미 시작된 두 학습 프로세스는 끝까지 완료됐다.

## 확정 규칙의 실제 실행 결과

두 모델 모두 공식 train의 128개 episode × 8 targets = **1,024개 학습 label을 1 epoch 동안 모두 처리**했다. 미도달 학습 label은 0개이며, 시퀀스 전체 BPTT를 사용했다. 원본 weight는 hash 검사에서 불변이고 LoRA·entry/memory·wrap·exit parameter는 모두 실제로 업데이트됐다.

| 항목 | 0.5B | 7B |
| --- | ---: | ---: |
| 처리 학습 label | 1,024/1,024 | 1,024/1,024 |
| Optimizer updates | 64 | 64 |
| 선택된 checkpoint update | 32 | 64 |
| CE를 계산한 실제 block 방문 | 24,653 | 28,614 |
| 학습 중 직접 맞힌 label | 87 | 385 |
| 학습 중 정답 공급으로 넘긴 label | 937 | 639 |
| 기록된 학습 단계 시간 | 88.59초 | 162.19초 |
| 사용 물리 GPU | 1 | 3 |

학습과 같은 누적 iteration/실패 후 정답 공급 규칙으로, 공식 test의 동일한 16 episodes·128 targets를 평가했다.

| 모델 | 직접 정답 학습 전 → 후 | 직접 정답률 학습 전 → 후 | 강제 공급 target 학습 전 → 후 |
| --- | ---: | ---: | ---: |
| 0.5B | 9 → 9 /128 | 7.03% → 7.03% | 119 → 119 |
| 7B | 46 → 54 /128 | 35.94% → 42.19% | 82 → 74 |

이 표는 실패 이후 정답을 공급한 상태에서도 다음 label 평가를 계속한 결과다. 강제 공급 없는 자유 생성 완주율과 구분한다. 모든 token을 강제 공급 없이 맞힌 episode는 두 모델의 전후 모두 0/16이다.

- [0.5B 학습 전후 보고서](results/training_revision_05b/comparison.md)
- [7B 학습 전후 보고서](results/training_revision_7b/comparison.md)
- [0.5B 학습 기록](results/training_revision_05b/training/done.json)
- [7B 학습 기록](results/training_revision_7b/training/done.json)

정답을 입력으로 강제 공급하지 않는 loop는 별도로, 각 모델의 같은 2개 test episode를 조건별 5초 관측했다. Token별 방문 한도는 없었다.

| 모델 | 직접 맞힌 token 학습 전 → 후 | 완주 episode 학습 전 → 후 |
| --- | ---: | ---: |
| 0.5B | 1 → 1 /16 | 0/2 → 0/2 |
| 7B | 1 → 2 /16 | 0/2 → 0/2 |

이 짧은 관측에서 완주에 필요한 루프 수는 미확인이다. 더 오래 돌려도 완주 불가능하다는 뜻은 아니다. Gate 판정에는 여전히 데이터 정답을 사용하는 oracle이며, 여기서 제거한 것은 실패 뒤의 정답 입력 강제 공급이다.

- [0.5B 순수 loop 검증](results/training_revision_05b/strict_loop_check/report.md)
- [7B 순수 loop 검증](results/training_revision_7b/strict_loop_check/report.md)

실제 학습 trace 전량 검사에서 동일-layer 정답 전환, 강제 전환 시 다음 label 입력과 layer 1 시작, global iteration 비초기화와 `iteration <= target_index+1`, 모든 1,024 label 도달을 확인했다. CPU 검사 7개에서 다음 token loss만으로 이전 실패 hidden state에 gradient가 도달하는 것도 검증했다.

## 공통 규칙

- 실제 방문한 각 head에서 현재 학습 label의 next-token 또는 reconstruction CE를 계산한다.
- 중간에 정답이면 바로 다음 token을 넣고 같은 layer부터 이어간다.
- 허용 iteration 안에 못 맞힌 token도 loss를 남긴다. 정답 입력을 공급해 다음 token 학습을 **layer 1부터** 계속한다. 시퀀스를 버리지 않는다.
- 마지막 token까지 모든 학습 label을 처리했는지 assert한다. `forced_targets`와 `unforced_hits`를 따로 센다.
- 기본 `bptt_visits: 0`은 시퀀스 전체 graph를 유지한다. 뒤쪽 token loss가 앞의 state·주입·wrap·Transformer LoRA로 흐른다. 한 시퀀스 안에서는 optimizer를 업데이트하지 않는다.
- 메모리 때문에 `bptt_visits`를 양수로 지정하면 해당 방문 수마다 gradient history만 끊는 truncated BPTT가 된다. Numeric state/cache는 유지한다.

## 명시적으로 정해야 하는 두 값

### 정답 정렬 `label_mode`

| 값 | 실제 입력과 정답 | 의미 |
| --- | --- | --- |
| `next_token` | 입력 A → 정답 B | 입력 앞 문맥 다음에 올 token을 예측 |
| `input_token` | 입력 A → 정답 A | 이미 입력된 token 자체를 복원 |

두 방식 모두 현재 label ID를 각 방문 head의 CE 정답으로 쓴다. 차이는 해당 label을 모델 입력에 미리 포함하는지다. `input_token`은 reconstruction 과제이며 다음-token 생성 성능으로 해석하지 않는다.

### Iteration 단위 `iteration_policy`

Target index i는 context를 제외한 학습 대상 sequence의 0-based index다.

| 값 | 제한 규칙 |
| --- | --- |
| `per_token` | token마다 iteration을 1로 초기화. i번 token에 최대 i+1회 layer 순회 허용 |
| `global` | episode 전체의 layer 순회 iteration을 누적. 현재 target i의 허용 범위는 global iteration <= i+1 |
| `advance_on_last` | 마지막 layer에서 못 맞히면 즉시 다음 token으로 교체 |

예를 들어 모두 틀리는 3-token 시퀀스는 `per_token`이면 각 token에 1/2/3회씩 총 6회 순회, `global`이면 각 1회씩 총 3회 순회한다. 성공 후 다음 token을 중간 layer에 넣으면 첫 순회는 그 layer부터 마지막까지인 부분 순회다.

다음 순회를 시작하면 허용 범위를 넘는 지점에서 현재 token 탐색을 마치고 정답을 공급한다. `force_next` 전환은 이전 hidden state를 wrap 모듈로 전달한 뒤 다음 입력 token을 layer 1에 주입한다. 다음 token의 loss도 이 연결을 따라 역전파된다.

## 구성 파일과 실행기

- 실행기: `run_training_revision.py`
- 0.5B: `configs/training_revision_05b.yaml`
- 7B: `configs/training_revision_7b.yaml`

사용자 답변에 따라 두 설정 모두 `label_mode: next_token`, `iteration_policy: global`로 확정했다. 0번 target은 누적 iteration 1까지만 허용한다. 1번 target은 누적 iteration 2까지 허용하며, 이미 지난 iteration을 0으로 초기화하지 않는다. 다음 순회가 index+1을 초과하게 되면 현재 token의 탐색을 마치고 정답 입력을 공급해 다음 token 학습을 계속한다.

다른 기본값은 context 32 / continuation 8 tokens, train 128 episodes, validation 8, test 16, rank 8 LoRA, LR 1e-4, 1 epoch다. 기존 결과는 보존하고 `runs/training_revision_05b/` 또는 `runs/training_revision_7b/`에 새로 저장한다. 허용 GPU는 물리 1·2·3뿐이다.

`--stage prepare/before/train/after/compare` 및 전체 순차 실행을 지원한다. 실제 선택한 설정은 결과 폴더의 `resolved_config.yaml`에 고정한다.

```bash
python run_training_revision.py --config configs/training_revision_05b.yaml --gpu 1
python run_training_revision.py --config configs/training_revision_7b.yaml --gpu 3
```

각 명령이 데이터 준비, 무학습 평가, 새 경로 학습, 학습 후 평가, 비교 보고서를 모두 수행한다. 결과는 각각 `runs/training_revision_05b/comparison.md`, `runs/training_revision_7b/comparison.md`다. 완료된 단계는 동일 명령에 `--resume`을 붙여 재사용한다.

## 결과 해석

`processed_targets`는 학습 label 처리 수이며 생성 성공 수가 아니다. 예산을 소진해 정답을 공급한 target도 포함한다. 직접 맞힌 것은 `unforced_hits`, 공급받은 것은 `forced_targets`다. 강제 공급 없이 모두 맞힌 시퀀스 비율은 `all_correct_without_forcing_fraction`으로 별도 저장한다.

Validation에서는 직접 맞힌 비율을 우선하고, 동률이면 방문 head의 평균 CE가 낮은 checkpoint를 고른다. Test는 선택에 사용하지 않는다.

새 학습은 기존과 같은 `RoutedNetwork` architecture를 쓴다. Transformer q/v LoRA, token entry/memory, wrap, exit scale을 학습하고 원본 weight는 고정한다. Checkpoint에는 architecture source hash와 새 trainer source hash를 모두 기록한다.

`next_token`으로 학습한 checkpoint는 기존 loop pilot의 `source_experiment_config`를 새 결과의 `resolved_config.yaml`로 지정하여, 강제 공급 없는 실제 루프 평가에 사용할 수 있다. `input_token` 학습 결과를 그 다음-token 평가와 혼동하지 않는다.

새 학습 모델로 정답 강제 공급 없이 완주 루프를 관측하는 명령:

```bash
python run_loop_pilot_tokens.py --config configs/loop_pilot_revision_05b.yaml --gpu 1
python run_loop_pilot_tokens.py --config configs/loop_pilot_revision_7b.yaml --gpu 3
```

이는 재학습 명령이 아니라 새 checkpoint의 strict loop 평가다. Token별 방문 한도는 없고 기본적으로 episode당 60초 관측 후 미완주를 기록한다. 시간 제한 없이 학습 후 시퀀스 하나를 기다리려면 `--condition after --episodes 1 --max-seconds 0 --output-dir runs/loop_pilot_revision_05b_wait`와 같이 별도 결과 폴더를 지정한다.

## 수행한 기술 검증

- CPU 검사 7개 통과: 실패 이후 모든 label 도달, i+1 per-token/global 차이와 비초기화, 성공 시 같은 layer 입력, 두 정답 정렬, 다음 token loss를 통한 이전 실패 state의 실제 tiny Transformer gradient.
- 실제 0.5B에서 두 해석의 최소 전체 흐름을 별도 실행했다. 각 train은 4 episodes × 3 targets = 12개 label이다. 본 학습의 최종 해석을 선택하기 위한 성능 실험은 아니다.

| 기술 검증 | 처리한 학습 label | 방문 head loss | 학습 중 직접 hit | 강제 공급 |
| --- | ---: | ---: | ---: | ---: |
| next_token + global | 12/12 | 289 | 1 | 11 |
| input_token + per_token | 12/12 | 576 | 0 | 12 |

- [next-token 분기 기술 검증](results/training_revision_check_next_global/comparison.md)
- [input-token 분기 기술 검증](results/training_revision_check_input_per_token/comparison.md)

둘 다 전체 시퀀스 BPTT, 실제 optimizer 업데이트, checkpoint 저장·복원, 학습 전후 비교 및 원본 weight 불변 검사를 통과했다. 기존 `routed_lm/`와 `loop_pilot/` 코드는 변경하지 않았다.
