# Layer별 토큰 예측과 순환형 생성 연구 목표

**제출용 연구계획서 재작성:** [Word 문서](proposal/LLM_Inference_Acceleration_Research_Proposal.docx), [PDF](proposal/LLM_Inference_Acceleration_Research_Proposal.pdf). 사용자 지시에 따라 readout adapter 학습 결과는 제안서에서 제외했다. 완료된 예비 근거는 무학습 모델의 최초 정답 깊이 21.30층과 coverage 52.77% 관측이다. 실제 중간 토큰 주입·순환 경로에서 Transformer 본체와 전환 모듈을 공동 학습하는 방법은 후속 검증 계획으로 명시했다. 기존 학습 결과는 실험 이력으로만 보존한다.

**확대 실행 완료 — 2026-11-03 확인:** 0.5B가 GPU 1·2·3에서 3 epochs, 393,216 episodes, 누적 3,145,728 targets를 모두 처리했다. 직접 정답 7.95%→30.97%, 전체 평균 block 방문 24.07→22.95. 첫 target 정답 비율은 42.48%→27.73%로 하락했고, 강제 정답 공급 없는 8-token 완주는 0→2/1,024였다. 정답을 아는 분기·실패 후 정답 공급 평가이며 원본 weight는 동결했다. 7B 확대 완료 기록은 없다. [상세 결과와 해석](PARALLEL_TRAINING.md).

**데이터 규모 이력:** 초기 새 학습은 1,024 target의 동작 검증이었다. 이후 0.5B는 전체 WikiText-2 train 2,517,232 Qwen tokens 중 1,048,576개 정답 위치를 추출한 확대 학습을 3 epochs 완료했다. [전체 데이터 실측](DATASET_SCALE.md), [확대 결과](PARALLEL_TRAINING.md).

**확정된 새 학습:** 입력 A → 정답 B의 next-token 학습, 시퀀스 전체 누적 iteration을 현재 target index+1과 비교한다. 예산 소진 시 해당 label의 loss를 남기고 정답 입력으로 다음 token 학습을 계속한다. 이 규칙으로 별도 0.5B·7B 학습 전후 비교를 진행한다. [구현·실행 명령](TRAINING_REVISION.md).

**실행 완료:** 두 모델 모두 1,024개 학습 label 전체 처리, 1 epoch, 64 optimizer updates. 새 평가 규칙에서 직접 정답 수는 0.5B 9→9/128, 7B 46→54/128. 정답 입력 강제 공급 없는 짧은 loop 관측은 두 모델 모두 완주가 없어 필요 loop 수 미확인. 상세 해석과 결과 링크는 위 학습 문서를 기준으로 한다.

**현재 추가 실험:** token별 방문 한도 없이 동일한 시퀀스·상태를 유지하며 완주까지 필요한 L→1 루프 수를 측정한다. 기존 학습 전·후 checkpoint를 재사용하며, 시간 내 미완주는 필요 루프 수 미확인으로 처리한다. [Loop pilot 구성](LOOP_PILOT.md).

**2026-11-02 정정:** readout-only 학습은 사용자 원안의 학습·평가를 대체하지 않는다. 현재 우선 실험은 **정답 시 같은 layer에 다음 token을 주입하고, 실패 시 상태를 이어 다음 layer 및 L→1로 진행하는 실제 경로의 학습 전후 비교**다. 새 구현·명령·state/cache 규약은 [ROUTED_EXPERIMENT.md](ROUTED_EXPERIMENT.md)를 기준으로 한다. 기존 E01 수치는 readout 분석 결과로 보존하며 실제 순환 경로의 성능으로 해석하지 않는다.

**기존 학습 checkpoint의 계산 예산:** token당 최대 **96회 block 방문**으로 학습한 `configs/routed_05b_v96.yaml`, `configs/routed_7b_v96.yaml`을 보존한다. **현재 완주 루프 수 pilot의 평가에서는 이 token별 제한을 해제한다.** 기존 결과는 `runs/*_v96/`, 새 pilot은 `runs/loop_pilot_*/`에 따로 저장한다.

작성일: 2026-11-01 (Asia/Seoul)  
상태: 0.5B/7B 소규모 실행 검증 및 0.5B 확대 학습·test 비교 완료. 7B 확대 학습은 완료 기록 없음. 최신 확대 결과는 [PARALLEL_TRAINING.md](PARALLEL_TRAINING.md), 초기 검증 이력은 [VALIDATION.md](VALIDATION.md).  
첫 실험: [EXPERIMENT_01_DESIGN.md](EXPERIMENT_01_DESIGN.md)

## 1. 사용자가 제안한 목표

기존 decoder-only Transformer는 일반적인 autoregressive decoding에서 새 토큰마다 첫 block부터 마지막 block까지 계산한다. 이 연구는 중간 block에서도 다음 토큰을 예측하고, 필요한 계산량을 토큰별로 조절하여 생성 비용을 줄이는 것을 목표로 한다.

KV cache를 사용하는 표준 decoding은 전체 prefix를 매번 재계산하지 않는다. 줄이려는 비용은 주로 새 입력 토큰이 지나가는 block 계산이며, prompt의 prefill 비용은 따로 고려한다.

원래 아이디어의 핵심은 다음과 같다.

1. Transformer의 각 block 뒤에 LM Head를 통한 예측 지점을 둔다.
2. 중간 layer에서 올바른 다음 토큰을 예측하면 그 토큰을 입력에 추가한다.
3. 다음 토큰의 계산은 방금 성공한 layer에서 시작한다.
4. 정답이 나오지 않으면 더 깊은 layer로 진행한다.
5. 마지막 layer에서도 실패하면 첫 layer로 돌아가 예측을 이어간다.
6. 이런 경로에서도 올바르게 생성하도록 중간 예측과 순환 동작을 학습한다.

여기서 **성공한 layer에서 다음 토큰을 시작하는 동작**과 **마지막 layer에서 첫 layer로 돌아가는 동작**을 장기 목표로 유지한다. 새 토큰마다 첫 layer로 돌아가는 일반적인 early exit만 구현해서 이 아이디어를 검증했다고 판단하지 않는다.

## 2. 첫 실험의 질문

> 기존 모델의 정상적인 forward 경로에서 각 layer의 다음 토큰 예측을 관측하면, 올바른 토큰이 최초로 나타나는 깊이는 평균 얼마이며, 얼마나 자주 끝까지 나타나지 않는가?

이 질문을 먼저 해결한다. 첫 실험은 전체 layer를 계산하는 관측 실험이다. 이후 layer를 실제로 생략하거나 순환하지 않는다. 따라서 결과는 중간 예측을 이용할 여지에 관한 증거이며, 추론 가속이나 순환 구조의 성능을 입증하는 결과는 아니다.

사용자 요청에 따라 **학습 전·후 비교를 필수 구성**으로 한다. 중간 hidden state가 최종 LM Head의 입력 공간에 맞춰져 있지 않을 수 있으므로, backbone을 고정하고 layer별 readout adapter만 학습한 조건을 같은 평가 데이터에서 비교한다. 실제 next-token CE를 주 학습 목표로 두고, 최종 분포를 모방하는 KL 학습은 선택 ablation으로 둔다. Affine readout 구조의 참고 근거는 [Tuned Lens 논문](https://arxiv.org/abs/2303.08112)과 [저자 구현](https://github.com/AlignmentResearch/tuned-lens)에 있으며, 주 비교의 CE 학습 목표는 이번 실험의 설계 선택이다. 학습 없이 읽은 중간 예측만으로 아이디어를 기각하지 않는다.

## 3. 정답의 두 가지 정의

| 이름 | 비교 대상 | 답할 수 있는 질문 |
| --- | --- | --- |
| Reference 정답 | 데이터에 실제 기록된 다음 token ID | 기록된 다음 토큰을 얼마나 일찍 맞히는가? |
| Teacher 일치 | 같은 prefix에서 최종 layer가 greedy하게 고른 token ID | 원래 모델의 출력을 얼마나 일찍 재현하는가? |

두 지표를 항상 따로 기록한다. Teacher 일치는 실제 정답 보장이 아니다. Reference의 한 토큰과 다르더라도 자연어로 가능한 다른 표현일 수 있다. 이번 실험은 토큰 예측 실험이며, 문장·수학 문제·코드의 의미적 정답 평가는 후속 단계로 둔다.

## 4. 성공 여부를 결정하는 세 문제

### 4.1. 중간 layer에 충분한 예측 정보가 있는가?

최초 정답 깊이뿐 아니라 이후 모든 layer에서 정답이 유지되기 시작하는 깊이, layer별 정확도, 마지막까지 실패하는 비율을 측정한다. 중간에 맞았다가 뒤에서 틀릴 수도 있으므로 정확도가 깊이에 따라 단조롭게 좋아진다고 가정하지 않는다.

### 4.2. 실제 추론에서 성공을 알아낼 수 있는가?

실험에서는 정답 데이터를 보고 최초 정답 layer를 찾는 oracle 평가가 가능하다. 실제 생성에서는 정답을 알 수 없으므로 별도 exit 판정기가 필요하다. Confidence, 인접 layer의 예측 일치, 학습된 gate 등을 validation에서 보정하고, 오답 상태로 exit하는 비율을 test에서 평가해야 한다. 이런 문제는 [CALM](https://arxiv.org/abs/2207.07061)에서도 다룬다.

최종 layer의 결과로 검증하는 방식은 가능하지만 그 계산 비용도 포함해야 한다. Speculative 검증을 이용한다면 그에 맞는 별도 decoding 알고리즘이 필요하다. [LayerSkip](https://arxiv.org/abs/2404.16710)은 중간 예측 학습과 self-speculative decoding의 비교 기준이다.

### 4.3. 성공한 layer에서 다음 토큰을 계속 계산할 수 있는가?

기존 block의 입력 차원이 같다고 해서 raw token embedding을 어떤 깊이의 block에 넣어도 정상적으로 동작하는 것은 아니다. 다음 세 요소를 함께 설계하고 학습해야 한다.

- **시작 표현:** 새 토큰 embedding을 해당 깊이의 입력 공간으로 옮기는 layer별 adapter, 또는 문맥 상태와 결합하는 recurrent 입력 모듈.
- **문맥 접근:** 방문하지 않은 layer에는 과거 토큰의 K/V가 없다. 재계산, 상태 전달, 별도 recurrent memory 중 어떤 방식을 쓸지 정해야 한다.
- **재순환 상태:** 같은 prefix를 같은 embedding에서 다시 계산하면 deterministic inference에서는 같은 결과가 반복된다. 재순환이 의미 있으려면 이전 pass의 hidden state나 memory를 다음 pass로 전달해야 한다.

Sampling을 다시 해서 다른 토큰을 얻는 것은 별도 baseline으로 취급한다. 이것만으로 계산을 더 하면 추론이 개선된다고 해석하지 않는다.

## 5. 장기 구조의 잠정 동작 규약

사용자의 '거기서부터'는 우선 **성공한 block B_l을 다음 토큰에도 다시 실행하는 것**으로 해석한다. B_(l+1)부터 시작하는 해석은 별도 ablation으로 남긴다. 첫 실험의 결과는 이 선택에 의존하지 않는다.

```text
상태: 확정된 prefix, 예측할 token 위치 t, 현재 layer l,
      현재 hidden state, recurrent memory, layer별 KV 유효 범위, 방문 횟수

현재 block B_l 실행 → LM Head로 token 후보 예측 → gate 판정
  accept:
    후보를 prefix에 추가
    t를 1 증가
    새 embedding과 memory로 B_l의 다음 입력 생성
    다음 예측도 B_l부터 시작
  reject and l < L:
    현재 hidden state를 B_(l+1)에 전달
  reject and l == L:
    wrap adapter로 현재 hidden state/memory를 갱신
    같은 token 위치 t의 예측을 B_1에서 계속
```

이 규약은 후속 구현의 출발점이며 검증된 모델 구조가 아니다. 순환 시의 cache 중복·갱신, attention mask와 position ID의 의미를 먼저 명세해야 한다. 동일 token을 재방문할 때 일반적인 append-only KV cache를 그대로 사용하지 않는다. `max_block_visits_per_token` 같은 계산 예산과 예산 초과 시 fallback도 필요하다. 더 많이 순환하면 반드시 정답에 도달한다는 보장은 없다.

## 6. 단계별 연구 계획

| 단계 | 구현 및 평가 | 다음 단계 판단 근거 |
| --- | --- | --- |
| E01-A, 필수 | 고정 모델 + identity readout의 학습 전 관측 | 최초·안정 정답 깊이와 실패율 |
| E01-B / B-CE, 필수 | 고정 backbone + CE로 학습된 readout adapter | 동일 평가 token에서 학습 전후 변화량 |
| E02 | Early-exit용 공동 학습, 실제 gate 평가 | 정확도 손실 대비 계산 절감, 최종 layer 품질 유지 |
| E03 | 성공한 layer에서 새 토큰을 시작하는 oracle 경로 | 시작 표현·문맥 상태·cache 규약의 실현 가능성 |
| E04 | 상태를 전달하는 L→1 재순환 학습 | 추가 방문의 정답 개선량과 비용 |
| E05 | 학습 gate를 포함한 실제 autoregressive 생성 | 품질, tokens/s, 지연, VRAM, 총 block 방문 수 |

E03/E04에서는 정상 경로에서 얻은 E01 hidden state를 그대로 대신 사용하지 않는다. 입력 경로가 달라지면 중간 상태와 예측도 달라지므로 경로 자체를 실행해 평가한다.

E01은 A/B 두 조건과 학습 전후 비교 보고서가 모두 완성되어야 완료다. Backbone 전체의 공동 학습과 순환 경로 학습은 후속 단계에서 별도로 평가한다.

## 7. 연구 판단 원칙

- '평균 몇 layer인가'는 성공 표본 평균과 전체 표본의 실패 포함 비용을 함께 제시한다.
- Oracle의 최소 깊이는 배포 가능한 stopping rule이 아니다.
- 적은 layer 수는 곧바로 속도 향상이 아니다. LM Head, adapter, gate, cache 처리, prefill, 재순환 비용까지 측정한다.
- 독립적인 거대 vocabulary projection을 layer 수만큼 복제하는 방식을 기본값으로 삼지 않는다. 공유 head와 작은 adapter로 시작한다.
- 최종 모델과의 일치율, reference 정확도, 실제 task 품질을 서로 바꾸어 표현하지 않는다.
- 실제 수치가 나오기 전에는 기대 speedup이나 정답 도달 깊이를 결과처럼 쓰지 않는다.

## 8. 현재 결정과 남은 선택

현재 결정: 실험을 독립 디렉터리 `/home/hyoungchan38/layerwise-lm-experiments`에서 시작하고 기존 RCA/PPO 작업과 분리한다. E01은 next-token top-1을 기본으로 하며, 무학습 baseline과 CE로 readout을 학습한 조건을 모두 수행해 같은 test에서 비교한다. 학습 전후 성공 subset이 바뀌므로 공통-hit token의 깊이 변화, 실패→성공/성공→실패 비율, 실패 포함 계산량도 기록한다.

첫 실험의 모델·데이터: `Qwen/Qwen2.5-0.5B-Instruct`로 pilot을 수행하고 `Qwen/Qwen2.5-7B-Instruct`로 본 평가를 수행한다. 데이터는 `Salesforce/wikitext`의 `wikitext-2-raw-v1`이며, train은 readout 학습, validation은 pilot·설정 선택, test는 고정 평가에 사용한다.

사용자 지정 자원 제한: **물리 GPU 1·2·3만 사용한다. GPU 0은 사용하지 않는다.** 이 제한은 pilot, 추론, 학습, 후속 실험에 계속 적용한다.

미확정: 최종 적용 모델·도메인, 새 토큰 시작 모듈, 순환 memory/cache 구조, gate의 허용 오류율, 실제 task 품질 기준. 첫 실험은 이 선택 없이 진행할 수 있도록 설계한다.

## 9. 기록

- 2026-11-01: 사용자 원안, 첫 관측 실험, 후속 학습·순환 검증 단계를 분리하여 저장. GPU와 로컬 모델 config만 확인했으며 모델 추론·학습은 실행하지 않음.
- 2026-11-01: 사용자 지시에 따라 물리 GPU 1·2·3만 사용하도록 제한을 기록하고 첫 실험의 모델·데이터를 명시함.
- 2026-11-01: 사용자 지시에 따라 학습 전·후 비교를 필수 단계로 변경. 주 학습 조건을 reference next-token CE 기반 readout adapter 학습으로 정하고, 같은 test token의 paired 비교를 완료 기준에 포함함.
- 2026-11-01: 실행 코드·설정·명령 안내 구현. CPU 단위 검사 6개와 실제 0.5B/7B의 소규모 학습 전후 전체 흐름을 검증. 본 실험 데이터 고정 subset을 준비했으며 본 규모 학습·test 평가는 사용자 실행 명령으로 제공함.
