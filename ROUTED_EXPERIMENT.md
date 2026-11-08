# 원안 실험: 정답 layer에서 다음 token을 주입하며 이어가는 학습

**완주에 필요한 루프 수 관측:** 기존 모델을 token별 제한 없이 계속 실행하는 새 pilot은 [LOOP_PILOT.md](LOOP_PILOT.md)와 `run_loop_pilot.py`를 사용한다. 아래 96회는 기존 학습·제한 평가의 설정이다.

실제 수행 범위와 결과는 [ROUTED_VALIDATION.md](ROUTED_VALIDATION.md)에 기록한다.

2026-11-02. 사용자의 원안은 실제 계산 경로를 바꾸는 실험이다. 기존 `run.py`의 frozen-backbone readout 측정 결과는 이 동작을 검증한 결과가 아니다. 이 문서는 새 `run_routed.py` 구현의 규칙과 명령을 설명한다.

## 실행 규칙

각 episode는 원문의 context와 이어지는 target token sequence로 구성된다. 첫 target을 예측할 때 context의 마지막 token을 layer 1부터 계산한다.

```text
현재 input token을 block l에 통과 → LM Head → 실제 다음 token과 비교

맞힘:
  예측한 token을 즉시 입력에 추가
  새 token embedding과 직전 hidden memory를 결합
  같은 block l부터 다음 token 예측 시작

틀림, l < L:
  같은 input token의 hidden state를 block l+1로 전달

틀림, l == L:
  현재 hidden state를 wrap 모듈로 변환
  같은 target의 예측을 block 1에서 계속
```

예를 들어 `token a → layer 8에서 b 정답 → token b를 layer 8에 입력 → layer 9에서 c 정답 → token c를 layer 9에 입력`이 실제 실행된다. 성공한 token에 대해 남은 layer를 계산하지 않는다. 같은 layer를 다음 token에도 다시 실행하는 해석으로 구현했다.

정답 판정은 **실험 데이터의 다음 token label을 사용하는 oracle**이다. Test에서도 이 label은 accept/continue 분기에 사용한다. Test label을 optimizer loss로 역전파하지는 않는다. 이는 경로의 가능성과 계산량을 연구하기 위한 oracle 실험이며, 정답을 모르는 실제 서비스용 gate는 포함하지 않는다.

## 상태와 KV cache

새 token 입력은 `embedding + layer별 entry delta + gated previous-hidden memory`다. 학습되는 entry/memory 모듈이 직전 exit 상태를 새 token에 전달한다. Wrap도 직전 hidden state를 전달하며, 동일 prefix를 초기 embedding에서 다시 실행하는 방식으로 reset하지 않는다.

각 layer는 **실제로 방문한 input-token 위치의 K/V만** 저장한다. 방문하지 않은 layer의 과거 K/V를 만들기 위해 다른 block을 몰래 계산하지 않는다. 같은 token이 재순환으로 같은 layer를 다시 방문하면 해당 token 위치의 K/V를 새 값으로 교체하며, 중복 append하지 않는다. Token 위치는 실제 입력 위치이고 wrap 횟수로 증가시키지 않는다. Attention은 현재까지 방문한 위치만 볼 수 있다.

이 sparse cache와 recurrent hidden-memory의 결합은 이번 프로토타입의 구체적인 상태 설계다. Standard full-depth Transformer의 KV와 동일하다고 가정하지 않는다.

Episode 시작 시 **context의 마지막 token 전까지** prompt를 full-depth로 prefill한다. 이후 예측할 token의 계산은 위 경로만 사용한다. Prefill의 block×token 비용은 decode block 방문 수와 분리해 기록한다.

## 무엇을 학습하는가

- 실제 방문한 Transformer block의 **q/v projection LoRA**
- 새 token을 exit layer에 넣는 **entry 및 memory 모듈**과 carry gate
- 마지막 layer 상태를 첫 layer로 넘기는 **wrap 모듈**
- 각 layer의 readout scale

기존 base weight/embedding/final norm/공유 vocabulary LM Head는 고정한다. Transformer의 유효 계산은 학습된 LoRA로 바뀐다. 기존 readout-only 실험과 달리 **실제 재진입·재순환 경로를 통과하는 상태에서 학습**한다.

Loss는 각 실제 방문 block에서 계산한 원문 next-token CE다. 경로 선택은 argmax와 oracle label로 결정하며, 선택 자체를 미분하거나 RL로 학습하지 않는다. 방문한 상태의 연속 계산을 truncated BPTT로 역전파한다. BPTT 경계에서 수치 상태와 cache는 유지하고 gradient history만 detach한다. 새 token 주입/wrap을 detach 이후에 적용하여 그 모듈의 gradient가 다음 block loss로 이어지도록 한다.

## 실패와 계산 예산

사용자 지정에 따라 현재 실행용 `*_v96.yaml`은 **학습과 평가 모두 token당 최대 96회의 block 방문**을 사용한다. 이 한도에서 맞히지 못하면 **해당 episode를 종료**한다. 정답을 강제로 주입해 다음 target으로 진행하지 않는다. 이후 target은 `unreached`로 기록한다. 기존 0.5B 48회·7B 56회 설정과 결과는 보존했으며, 96회 결과는 별도 `*_v96` 디렉터리에 저장한다.

학습도 같은 strict 경로를 사용한다. 따라서 설정의 `episodes × continuation_tokens`는 요청된 target slot 수이며, 모두 학습·평가에 도달했다고 해석하면 안 된다. 실제 accepted target, budget 실패, unreached, 방문 block 수는 trace와 보고서에서 확인해야 한다.

## 데이터와 비교

`Salesforce/wikitext`, `wikitext-2-raw-v1`, 고정 revision 사용. **공식 train / validation / test를 각각 학습 / checkpoint 선택 / 비교 평가에 사용한다.** 기존 0.5B pilot처럼 validation을 평가용으로 재분할하지 않는다.

문서를 먼저 섞고, 문서별로 순환 선택하면서 random offset에서 `context + continuation`을 추출한다. 같은 split에서 target의 `(document_id, absolute_position)`이 중복되지 않게 한다. 문서 순환 샘플링으로 긴 문서 하나가 표본을 독점하지 않도록 한다. 학습 전후는 동일한 episode manifest를 사용하지만 모델 상태가 달라져 경로와 도달 범위는 달라질 수 있다.

Checkpoint는 validation의 전체 요청 target 대비 resolved fraction을 우선하고, 동률이면 방문한 block의 평균 CE가 낮은 것을 고른다. Test로 checkpoint를 고르지 않는다.

핵심 비교 지표:

- 요청 target 전체 중 해결한 비율, episode 완료율
- 예산 실패 target 수와 아직 도달하지 않은 target 수
- 실제 같은-layer token 주입 횟수, 실제 L→1 wrap 횟수
- 성공 token별 **실제로 방문한 block 수**와 exit layer 번호
- 실패 비용을 포함한 resolved target당 block 방문 수
- 같은 target을 양쪽 모두 해결한 경우의 방문 수 변화

Wrap 이후 layer 3에서 맞혔다면 비용은 단순히 3이 아니다. 이전 방문을 모두 센다. 실패로 일찍 끝난 실행의 적은 총 방문 수를 speedup으로 해석하지 않는다. Oracle 분기와 vocabulary head 호출 비용까지 있어 실제 서비스 가속은 별도 측정이 필요하다.

## 명령

```bash
source /home/hyoungchan38/anaconda3/etc/profile.d/conda.sh
conda activate ppo_gpu
cd /home/hyoungchan38/layerwise-lm-experiments
```

실제 0.5B의 최소 전체 흐름 검증:

```bash
python run_routed.py --config configs/routed_smoke_05b_v96.yaml --gpu 1
```

0.5B의 실제 경로 학습 전후 비교:

```bash
python run_routed.py --config configs/routed_05b_v96.yaml --gpu 1
```

기본 설정은 train 128 episodes, validation 8, test 16, context 32 tokens, continuation 8 tokens, 1 epoch다. Nominal target slot은 각각 1,024 / 64 / 128이며, 예산 실패 이후 slot은 unreached다.

7B의 같은 실험:

```bash
python run_routed.py --config configs/routed_7b_v96.yaml --gpu 3
```

필요하면 `--stage prepare`, `before`, `train`, `after`, `compare`를 순서대로 실행한다. 예를 들어 0.5B는 다음과 같다.

```bash
python run_routed.py --config configs/routed_05b_v96.yaml --stage prepare --gpu 1
python run_routed.py --config configs/routed_05b_v96.yaml --stage before --gpu 1
python run_routed.py --config configs/routed_05b_v96.yaml --stage train --gpu 1
python run_routed.py --config configs/routed_05b_v96.yaml --stage after --gpu 1
python run_routed.py --config configs/routed_05b_v96.yaml --stage compare --gpu 1
```

전체 실행과 단계별 실행 중 한 방식을 사용한다. GPU는 물리 번호 1·2·3만 허용한다. 완료된 단계 재사용은 아래와 같다. `--resume`은 완료 단계 재사용이며 미완료 학습은 처음부터 다시 시작한다.

```bash
python run_routed.py --config configs/routed_05b_v96.yaml --gpu 1 --resume
```

96회 설정으로 사용자가 0.5B·7B의 학습·평가를 실행 완료했다. 실제 처리량과 시간은 [검증 기록](ROUTED_VALIDATION.md)의 96회 예산 결과에 있다. 모델·데이터·epoch 등 다른 설정은 기존 실험과 같고, token 방문 예산과 결과 경로만 변경했다.

## 결과와 검증

96회 0.5B는 `runs/routed_05b_v96/comparison.md`, 7B는 `runs/routed_7b_v96/comparison.md`에 비교 보고서를 저장한다. 각 결과 폴더의 `before_train/trace.jsonl`과 `after_train/trace.jsonl`에 실제 block별 경로를 저장한다. `training/train_trace.jsonl`은 학습 경로이며, `training/done.json`에는 LoRA·entry/memory·wrap·exit 각 parameter군이 바뀐 크기도 기록한다. 기존 48회 결과는 `runs/routed_05b/`에 유지한다.

Trace의 `action=emit` 다음 행은 같은 layer에서 다음 target을 예측해야 한다. `action=wrap` 다음 행은 같은 target을 layer 1에서 다시 예측해야 한다. `budget_exhausted` 다음에는 해당 episode의 추가 block 방문이 없어야 한다.

```bash
python -m unittest discover -s tests_routed -v
```

검사는 정답 후 즉시 동일 block 재진입, 재순환 state 전달, 실패 후 강제 정답 주입 금지, KV 교체와 누락 위치 처리, 일반 full-depth prompt와 cache 수치 동등성, 실제 tiny Transformer에서 entry/wrap/LoRA의 gradient 전달을 포함한다.

기존 `run.py`와 원래 `src/layerwise_lm/`는 보존했다. 새 코드가 실행 중인 기존 readout 실험의 source hash를 바꾸지 않도록 별도 `routed_lm/`에 구현했다. 기존 readout 결과는 이 원안 실험의 결과로 합산하지 않는다.

Attention/cache API 근거: [Transformers cache 문서](https://huggingface.co/docs/transformers/main/en/kv_cache). Sparse visited-token cache와 recurrent 입력 모듈은 본 실험의 설계 선택이다.
