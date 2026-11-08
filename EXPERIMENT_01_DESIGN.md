# E01: 학습 전후 Layer별 다음 토큰 정답 도달 깊이 비교

**2026-11-02 정정:** 이 문서는 기존 readout-only 보조 실험 기록이다. 사용자 원안의 실제 token 주입·layer 재진입·순환 학습은 [ROUTED_EXPERIMENT.md](ROUTED_EXPERIMENT.md)와 새 `run_routed.py`를 기준으로 한다.

작성일: 2026-11-01 (Asia/Seoul)  
상태: Teacher-forced 학습 전후 비교 코드 구현 및 0.5B/7B 소규모 실행 검증 완료. 본 규모 학습·test 평가는 미실행. 실행 명령과 실제 파일 형식은 [README.md](README.md), 검증 기록은 [VALIDATION.md](VALIDATION.md)를 기준으로 한다. 아래 생성 prefix·KL ablation 등은 후속 확장 설계다.  
장기 목표: [RESEARCH_GOALS.md](RESEARCH_GOALS.md)

## 1. 이번 실험의 결과물

모델의 각 Transformer block 뒤에서 다음 토큰 분포를 읽고, 다음 질문에 답하는 재현 가능한 측정 도구를 만든다.

1. Reference token을 최초로 맞히는 layer의 평균·중앙값·분포는 무엇인가?
2. 전체 layer 중 한 번도 맞히지 못하는 token의 비율은 얼마인가?
3. 최초 정답 이후 다시 오답으로 바뀌는 경우는 얼마나 되는가?
4. 최종 layer의 예측과 일치하는 시점은 실제 reference 정답 시점과 얼마나 다른가?
5. 중간 readout을 학습하면 위 지표가 얼마나 달라지는가?

사용자 지정 필수 구성은 **① 추가 학습 없이 평가 → ② 중간 readout 학습 → ③ 같은 평가 데이터로 재평가 → ④ 학습 전후 비교**다. 학습은 baseline 성능에 관계없이 수행하며, 두 조건의 결과와 비교 보고서가 모두 있어야 이번 실험이 완료된다.

전체 forward를 완료한 뒤 통계를 낸다. 실제 조기 종료, 성공 layer에서 다음 토큰 시작, L→1 재순환은 E01의 구현 범위 밖이며 후속 실험에 포함한다. E01의 결과만으로 그 동작의 성공 가능성을 확정하지 않는다.

## 2. 확인한 환경과 기본 모델

로컬에서 읽기 전용으로 확인한 정보:

| 항목 | 확인 내용 |
| --- | --- |
| GPU | RTX 4090 중 **물리 GPU 1·2·3만 사용 가능. GPU 0은 사용 금지.** 사용 가능 GPU의 여유 VRAM은 실행 시 확인 |
| 기존 실험 | `baselines/rlhf_qwen`에 Qwen PPO 작업 존재; 이번 실험은 별도 폴더 사용 |
| Pilot 모델 | `Qwen/Qwen2.5-0.5B-Instruct`: 24 blocks, hidden 896, vocabulary 151,936 |
| 본 실험 모델 | `Qwen/Qwen2.5-7B-Instruct`: 28 blocks, hidden 3,584, vocabulary 152,064 |
| 캐시 | 위 두 모델의 cache 디렉터리와 config 존재; weight 전체의 가용성은 구현 시 확인 |

확인 당시 로컬 `refs/main` revision:

- 0.5B-Instruct: `7ae557604adf67be50417f59c2c2f167def9a775`
- 7B-Instruct: `a09a35458c702b33eeacc393d103063234e8bc28`

기본 진행은 **0.5B에서 학습 전후 pilot → 7B 학습 전 평가 → 7B readout 학습 → 동일 test의 학습 후 평가 → 비교 보고서**로 한다. 최초 설계에서는 모델·데이터 다운로드나 GPU 작업을 수행하지 않았으며, 현재 구현 검증의 실제 수행 범위는 README와 VALIDATION에 기록한다.

BF16, batch size 1, sequence length 512, 허용된 단일 GPU에서 시작한다. 7B weight 자체는 BF16 기준 약 14–16GB 규모지만 실제 peak VRAM은 hidden state, vocabulary logits와 실행 환경에 따라 달라진다. 작은 pilot으로 메모리·시간을 측정하고 부족하면 sequence length와 projection chunk를 줄인다. GPU 여러 장을 하나의 큰 메모리처럼 가정하지 않는다. 병렬화는 필요할 때 모델별·seed별 독립 실행에 우선 사용한다.

**GPU 제한은 모든 후속 실행에 적용한다.** 기본 배치는 pilot을 물리 GPU 1, 7B baseline을 물리 GPU 2, readout 학습을 물리 GPU 3에 배정하는 것이다. 실행 시 가용 상태에 따라 1·2·3 안에서 변경할 수 있으며 GPU 0으로 fallback하지 않는다. 실제 학습 메모리에 따라 1·2·3 내에서만 분산 구성을 검토한다.

프로세스를 시작하기 전에 물리 GPU index와 UUID를 확인하고 해당 UUID를 `CUDA_VISIBLE_DEVICES`에 지정한다. 프로세스 내부의 `cuda:0`은 제한 후 재번호가 붙은 첫 장치를 뜻하므로 물리 GPU 0과 혼동하지 않는다. 실행 manifest에 물리 index, UUID, 프로세스 내부 device의 매핑을 기록하고, 허용 목록 밖 장치가 있으면 실행을 중단한다.

Instruct 모델을 일반 텍스트에 적용한 결과와 chat completion 결과를 구분한다. 위 캐시 모델로 pilot을 시작하되, 일반 LM에 대한 결론을 넓히려면 같은 크기의 base 모델에서도 재현하는 후속 조건을 둔다. 기본 텍스트 평가는 chat template 없이 수행한다. 모델 ID와 실제 config를 결과에 저장한다.

## 3. 비교 조건

| 조건 | Backbone | 중간 readout | 의미 |
| --- | --- | --- | --- |
| A: no-train, 필수 | 고정 | Identity adapter + 최종 norm + 기존 LM Head 공유 | 추가 학습 없는 기준선 |
| B-CE: after-train, 필수 | 고정 | 학습된 layer별 affine adapter + 같은 norm/head | 실제 다음 token CE로 학습한 뒤 평가 |
| B-KL: distillation, 선택 | 고정 | B-CE와 같은 adapter 구조 + 공유 norm/head | 최종 모델 분포를 재현하는 별도 비교 조건 |
| C: deep supervision | 학습 | 여러 exit의 loss로 공동 학습 | E02에서 수행; A/B와 별도 checkpoint로 비교 |

**주 비교는 A 대 B-CE로 고정한다.** 실제 정답 token을 더 일찍 맞히는 것이 이번 목적이므로 학습 목표도 reference next-token CE로 둔다. B-KL은 추가 ablation이며 필수 학습 후 조건을 대신하지 않는다. 랜덤 초기화한 독립 LM Head의 성능을 무학습 기준선으로 사용하지 않는다.

A는 B와 동일한 adapter 구조에서 `A_l=0, b_l=0`인 identity 상태다. 수치적으로 기존 shared logit lens와 같아야 하며, B는 이 상태에서 시작하여 adapter만 학습한다. 모델 revision, tokenizer, norm/head, hidden 관측 지점, precision, 평가 token·prefix·mask를 모델별 A/B에서 고정한다. 학습 차이를 모델 교체나 평가 subset 변경과 섞지 않는다.

최종 layer는 A/B에서 원래 모델의 norm과 head를 그대로 사용하며 adapter는 identity로 고정한다. Backbone, embedding, norm, LM Head는 전부 freeze한다. 특히 0.5B는 input embedding과 output head weight가 묶여 있어 공유 head를 학습하면 embedding도 바뀔 수 있다.

이 설계는 [Tuned Lens](https://arxiv.org/abs/2303.08112)의 affine readout 학습을 참고한다. 중간 예측을 위한 backbone 공동 학습은 [LayerSkip](https://arxiv.org/abs/2404.16710)과 [CALM](https://arxiv.org/abs/2207.07061)을 E02의 비교 기준으로 삼는다. 사용자의 순환 구조와 이 방법들이 동일하다는 의미는 아니다.

## 4. 입력·정답·평가 모드

### 4.1. 주 실험: teacher-forced reference prefix

문서의 token sequence가 `x[0], ..., x[T-1]`이면, position `i`의 hidden state는 prefix `x[:i+1]`를 보고 **`x[i+1]`**을 예측한다.

```text
입력 위치:     0     1     2     ...  T-2
입력 token:    x0    x1    x2    ...  x[T-2]
정답 token:    x1    x2    x3    ...  x[T-1]
```

Causal mask를 사용한 한 번의 forward로 여러 position을 병렬 평가할 수 있다. 정답 token을 같은 위치의 hidden state에 노출하지 않는다. `hidden[:, :-1]` 대 `input_ids[:, 1:]`의 shift와 attention mask를 명시적으로 처리한다.

기본 데이터는 [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)의 `wikitext-2-raw-v1`. 공식 train/validation/test 분리를 유지한다. 이 데이터의 영어 텍스트 결과를 한국어·코드·수학으로 일반화하지 않는다. 해당 도메인은 출처와 분할이 고정된 JSONL을 같은 인터페이스로 추가한다.

전처리 규약:

- WikiText의 최상위 article heading을 기준으로 문서를 복원하고 내부 줄바꿈을 유지한다. section heading과 문서 경계 판별 규칙을 코드와 manifest에 기록한다.
- 서로 다른 문서를 separator 없이 이어 붙이지 않는다. 기본은 문서별 독립 시퀀스이며 BOS/EOS 추가 여부를 명시한다.
- 긴 문서는 512-token chunk로 처리하고 64-token prefix overlap을 준다. 각 chunk의 local **target token index** j가 `64 <= j < chunk_length`인 경우만 평가하고, 예측은 hidden index `j-1`에서 읽는다. 따라서 문서의 첫 64개 token은 context로만 사용하며, 뒤 chunk에서도 이미 평가한 overlap target을 다시 세지 않는다.
- 예를 들어 첫 chunk가 절대 token index 0..511이면 target 64..511을 평가한다. 다음 chunk는 448..959이며 target 512..959를 평가한다. Target 512는 두 번째 chunk의 hidden local index 63에서 예측한다. 이 규약으로 경계의 다음 token이 빠지는 off-by-one 오류를 막는다.
- 평가 target은 문서 내 절대 token 위치로 식별한다. Context가 chunk 경계에서 잘리는 조건임을 결과에 명시한다.
- padding, 문서 경계 예측, 인위적으로 넣은 special-token target은 제외한다. 자연스럽게 등장하는 EOS를 평가할지는 config로 고정한다.
- 기본 결과는 유효 token 수로 가중한 micro 평균, 문서별 평균의 macro 통계도 보조로 제공한다.

데이터 규모의 초기값:

| 용도 | 목표 규모 | 사용 규칙 |
| --- | --- | --- |
| Pilot | validation의 평가 token 8,192개 | 인덱스·norm·메모리 확인; 최종 수치로 사용하지 않음 |
| Pilot B-CE 학습 | train의 최대 32,768 유효 token | 0.5B에서 학습·저장·복원 경로 확인; 성능 결론에는 사용하지 않음 |
| 본 B-CE 학습 | train의 최대 1,000,000 유효 token | 7B 학습 필수; 부족하면 실제 수량 기록; train만 사용 |
| Hyperparameter 선택 | validation 문서의 50% | 문서 ID hash로 고정 분리 |
| Gate 보정 확장 | 나머지 validation 문서 50% | E02에서 사용; test와 분리 |
| 본 평가 | test의 최대 100,000 유효 token | 고정 seed로 문서를 고른 뒤 target cap 적용 |

최종 선택된 token ID, 문서 ID, 평가 mask, 순서와 데이터 fingerprint를 저장한다. A와 B는 같은 test token을 평가한다. 10만 token을 독립 표본 10만 개로 취급하지 않고, 문서 단위 bootstrap으로 95% CI를 계산한다. A/B 차이의 CI는 동일 문서를 함께 재표집하는 paired bootstrap으로 계산한다. 선택된 문서 수가 적으면 CI의 한계도 기록한다. Test는 checkpoint 선택·학습 중 early stopping에 사용하지 않으며, baseline 결과를 보고 test subset을 바꾸지 않는다.

### 4.2. 보조 실험: 실제 생성 prefix

Full-depth 원래 모델로 greedy 생성하면서 매 step의 layer별 예측을 관측한다. 기본 64 prompts × 최대 128 new tokens. 첫 step은 prompt의 마지막 유효 position, 이후 step은 새 입력 token의 마지막 position만 읽는다.

이 모드의 정답 기준은 기본적으로 teacher token이다. 생성 prefix가 reference에서 벗어난 뒤 원래 문서의 다음 token을 계속 '정답'으로 비교하지 않는다. Cached decoding에서는 모든 layer를 계속 계산하여 정상 KV cache를 유지한다. 이 모드도 실제 early exit가 아니다.

Teacher-forced 결과와 생성 prefix 결과를 서로 다른 run/mode로 보고한다. 첫 MVP는 4.1을 완성하고 4.2를 추가한다.

## 5. Layer 관측과 LM Head

Block 번호는 1부터 L까지, embedding은 layer 0으로 따로 둔다. `h_l(i)`를 **block l의 출력, 최종 norm 적용 전** residual state로 정의한다.

```text
h_0 = embedding(input_ids)
h_l = B_l(h_(l-1), causal_mask, position_ids)
z_l = lm_head(final_norm(h_l))
prediction_l = argmax(z_l)
```

LM Head는 논리적으로 각 block 뒤에서 호출하지만 weight는 공유한다. 각 layer마다 별도 vocabulary projection을 복제할 필요가 없다. 7B config 기준 하나의 BF16 `3584 × 152064` projection만 약 1.09GB이므로 독립 head 28개는 weight만 약 30.5GB이다. A의 추가 head parameter는 0개다.

모델별 adapter에서 `blocks`, `final_norm`, `lm_head` 경로를 제공한다. Qwen2 계열의 기준 경로는 `model.model.layers`, `model.model.norm`, `model.lm_head`이며 구현 시 설치된 Transformers 버전에서 검증한다. 참고 소스: [공식 Qwen2 구현](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py).

관측 구현은 각 decoder block의 forward hook으로 raw block 출력을 얻는 방식을 기본으로 한다. `output_hidden_states`를 쓸 경우 마지막 항목이 이미 final norm을 통과했는지 확인한다. 모델·버전마다 다를 수 있으므로 tuple index를 무조건 raw block 번호로 취급하지 않는다. norm을 두 번 적용하는 실수를 막기 위해 마지막 block의 raw hook 출력을 기준으로 일치 검사를 한다.

Hook은 원래 block 출력을 수정하지 않는다. 첫 구현은 `eval()`, `torch.inference_mode()`, 단일 장치, `torch.compile` 없이 검증한다. Output이 tensor인지 tuple인지도 adapter에서 처리한다.

## 6. 메모리와 실행 흐름

`[batch, sequence, layers, vocabulary]` 전체 logits를 보관하지 않는다. Teacher-forced 평가에서 block별 유효 position을 작은 chunk로 projection하고, 필요한 통계만 GPU에서 계산하여 CPU로 옮긴다.

```python
# 개념 의사코드. 실행 가능한 API가 아님.
with install_block_hooks(model, observer):
    for batch in dataloader:
        observer.begin(batch.target_ids, batch.eval_mask)
        output = model(**batch.model_inputs, use_cache=False)
        # hook마다: raw h_l → norm/head → 소규모 projection chunk
        # 저장: top-k ID, 정답 logprob, confidence, entropy 등
        observer.finish(final_logits=output.logits)
        reducer.update(observer.compact_records)
write_results(reducer)
```

기본 projection chunk는 유효 position 32개. 이때 7B vocabulary의 FP32 logits는 chunk당 약 19.5MB이다. BF16 matmul 후 log-softmax·통계는 FP32로 계산한다. 이 값은 projection tensor의 크기이며 peak VRAM 보장이 아니다.

기본 저장은 position×layer의 top-5 ID와 scalar 통계다. Teacher의 최종 top-1 ID는 전체 forward 후 알 수 있으므로 그때 compact prediction과 비교한다. Top-5 저장으로 teacher top-5 포함률도 계산할 수 있다. **Teacher 최종 token의 정확한 중간-layer logprob은 별도 projection 재실행 없이는 저장하지 못하므로 MVP 필수 지표로 넣지 않는다.** B-KL 학습은 별도 경로에서 teacher의 full distribution을 chunk별로 생성한다.

Normal final logits를 observer가 재사용할 수 있으면 마지막 projection 중복을 피한다. 최초 검증 때만 raw last-block→norm/head 경로와 원래 logits를 둘 다 계산한다. 구현에 따라 최종 logits가 sequence 전체로 생성될 수 있으므로 이것도 메모리 예산에 포함한다.

생성 모드에서는 full-depth KV cache를 정상적으로 유지하며 각 step의 마지막 유효 position만 관측한다. KV cache와 각 layer의 모든 hidden state를 디스크에 저장하는 방식은 기본값으로 쓰지 않는다.

## 7. 핵심 지표의 정확한 정의

유효 target position을 n, layer를 l=1..L, 예측 token을 `p[n,l]`, reference를 `y[n]`, 최종 예측을 `q[n]=p[n,L]`로 둔다. 조건 A/B 각각에서 reference·teacher 지표를 계산한다.

### 7.1. Layer별 품질

- Reference top-1: `A_ref(l) = mean(p[n,l] == y[n])`
- Teacher top-1 agreement: `A_teacher(l) = mean(p[n,l] == q[n])`
- Reference top-5 포함률, reference NLL, perplexity=`exp(mean NLL)`
- Layer별 entropy, maximum probability, top-1/top-2 margin
- Final-layer reference top-1 및 NLL: 전체 모델 baseline 품질

### 7.2. 최초 정답 깊이와 실패

Reference 기준 `D_ref(n) = min{l: p[n,l] == y[n]}`. 그런 layer가 없으면 `D_ref=null`로 저장한다. Teacher 기준 `D_teacher`도 동일하게 정의하되 target은 q[n]이다.

반드시 함께 보고할 값:

1. `coverage_ref = P(D_ref exists)`와 `no_hit_rate = 1 - coverage_ref`
2. 성공 token에 한정한 `mean(D_ref | hit)`, median, p90
3. 전체 token에 대한 최초 정답 누적률 `F_ref(l)=P(D_ref <= l)`
4. `mean(D_teacher)`, median, p90와 teacher 최초 일치 분포
5. 최종 layer가 reference를 맞힌 subset에서의 `mean(D_ref)`와 그 subset 크기
6. 중간에는 reference 정답이 있었으나 마지막에는 틀린 비율

Teacher 기준은 마지막 layer에서 반드시 일치하므로 coverage=1이어야 한다. Reference 기준 최초 정답은 마지막까지 없을 수 있다. 실패를 조용히 버린 평균만 제시하지 않는다. 시각화의 '미도달' bin은 독립 범주이며 `null`을 실제 layer L+1로 평균 내지 않는다.

### 7.3. 안정 정답 깊이와 flip

`S_ref(n) = min{l: 모든 j >= l에서 p[n,j] == y[n]}`. 최종 예측이 reference와 다르면 존재하지 않는다. `S_teacher`도 같은 방식으로 계산한다.

- 안정 깊이 평균과 존재 비율을 함께 보고한다.
- Reference 최초 hit 이후 한 번이라도 다시 오답이 되는 비율을 hit subset에서 측정한다.
- 인접 layer 간 top-1 변경 횟수와 correct→incorrect 전이 비율을 기록한다.

예시: `L=6`, reference 정답 여부가 `[0,1,0,1,1,1]`이면 `D_ref=2`, `S_ref=4`. 전부 0이면 둘 다 null. `[0,1,0,0,0,0]`이면 최초 hit는 2이지만 안정 정답 깊이는 null이다.

### 7.4. Oracle 계산량 지표

```text
C_ref(n) = D_ref(n), hit인 경우
           L,        끝까지 hit가 없는 경우

oracle_fullscan_mean = mean(C_ref)
oracle_block_saving = 1 - mean(C_ref) / L
teacher_oracle_block_saving = 1 - mean(D_teacher) / L
```

`C_ref`는 '정답을 알고 있다면 중간 정답에서 멈추고, 없으면 전체 깊이를 쓰는' 1-pass 비용이다. 실패 token을 맞히는 데 필요한 총 계산량이 아니다. LM Head·gate·KV 보완·prefill 비용을 포함하지 않으므로 end-to-end speedup으로 표현하지 않는다. 순환 이후 정답에 도달할 때까지의 비용도 E01로 추정할 수 없다.

전체 vocabulary head를 매 layer 계산하면 block을 적게 방문해도 느릴 수 있다. 지표 이름에 `oracle`과 `block`을 유지한다.

### 7.5. Gate 평가 확장

Validation에서만 probability/margin threshold 또는 작은 gate를 선택하고 test에서 risk-coverage와 accuracy-depth 곡선을 그린다. Gate 입력은 현재와 이미 방문한 과거 layer의 정보만 사용한다. Reference·최종 layer 결과·미래 layer는 정답 label로만 쓸 수 있다.

Gate의 false-exit rate, accepted coverage, 미수락 시 full-depth fallback을 포함한 평균 깊이와 전체 정확도를 함께 보고한다. 목표 허용 오류율은 후속 E02 설정값으로 정하며, E01에서는 threshold sweep을 관측 통계로만 저장할 수 있다.

## 8. B 조건의 readout 학습

각 중간 layer에서 `h'_l = h_l + A_l h_l + b_l`, `z'_l = lm_head(final_norm(h'_l))`로 둔다. Dense `A_l`과 bias를 0으로 초기화하면 A 조건에서 시작한다. 최종 layer는 adapter 없이 고정한다.

**필수 B-CE 학습:** 최종 layer를 제외한 모든 layer의 adapter를 실제 reference next-token 정답으로 학습한다. Layer별 loss에 같은 가중치를 주고, 유효 target mask만 사용한다.

```text
loss_l = cross_entropy(z'_l, reference_next_token)
loss = mean_{l=1..L-1}(loss_l)
```

학습 대상은 중간 readout adapter다. Backbone 전체 재학습은 E02의 별도 조건이며, 이번 A/B의 학습 후 결과는 '중간 readout 학습 효과'로 해석한다. Validation의 평균 중간-layer reference NLL이 가장 낮은 checkpoint를 선택하고 선택 규칙을 학습 전에 고정한다.

선택 확장 B-KL loss:

```text
teacher = stop_gradient(softmax(z_L / tau))
loss_l = tau^2 * KL(teacher || softmax(z'_l / tau))
loss = mean_l(loss_l)
```

선택 B-KL에서 `tau=1`로 시작한다. CE와 KL의 혼합을 추가하려면 별도 조건으로 명시하고 weight를 validation에서 선택한다. Teacher 분포를 학습한 B-KL이 reference 정확도를 최대화한다고 가정하지 않는다.

작은 pilot의 제안값은 AdamW, LR `1e-4`, 유효 batch 약 2,048 tokens/update, train token cap 1M, seed 42다. 이 값들은 성능이 확인된 설정이 아니며 짧은 validation으로 선택하고 기록한다. 주요 결과가 B에 의존하면 adapter 초기화/데이터 순서 seed 3개로 재현성을 확인한다.

Frozen backbone으로 batch의 hidden을 얻은 뒤 projection chunk와 layer별 loss를 순차 backward하여 전체 layer의 full-vocabulary autograd graph가 동시에 남지 않게 한다. Frozen head도 adapter로 향하는 gradient는 필요하므로 adapter 이후 projection을 `no_grad`로 감싸지 않는다. Backbone output은 detach한다.

7B에서 dense adapter 전체가 부담이면 projection chunk와 유효 microbatch를 줄이고, 필요하면 rank-64 저랭크 `A_l=U_l V_l` 구조로 전환한다. 이 경우 모든 중간 layer의 adapter를 학습하며 A도 같은 저랭크 구조의 identity 상태로 다시 검증한다. 저랭크는 dense tuned lens와 다른 조건이므로 구조 선택을 본 학습 전에 고정한다. Identity 초기화 시 U/V 양쪽을 모두 0으로 두어 학습이 멈추지 않도록 한쪽만 0으로 초기화한다. Adapter trainable parameter 수와 추론 비용을 결과에 포함한다.

### 8.1. 학습 전후 비교 순서

1. 모델별 고정 평가 manifest를 만들고 checksum을 저장한다.
2. A identity adapter로 학습 전 평가를 수행하여 `before_train/`에 저장한다.
3. Train split만으로 B-CE를 학습하고 validation으로 checkpoint를 선택한다.
4. 선택한 checkpoint를 새 프로세스에서 복원하고 동일 manifest로 `after_train/` 평가를 수행한다.
5. Target ID로 결과를 1:1 결합하고 누락·중복·prefix 차이가 없음을 검증한다.
6. 학습 전후 값, 변화량, paired CI를 담은 `comparison.csv`와 `comparison.md`를 생성한다.

최초 hit의 성공 표본 평균은 A와 B에서 모집단이 달라질 수 있다. 이를 보완하여 **양쪽 모두 hit인 공통 token의 깊이 변화**, A 실패→B 성공 비율, A 성공→B 실패 비율을 함께 보고한다. 전체 표본의 실패 포함 `mean(C_ref)` 차이도 필수다. Adapter가 최종 출력 경로를 바꾸지 않으므로 A/B의 teacher token과 final-layer reference 정확도·NLL은 수치 허용 오차 내에서 같아야 한다.

## 9. 제안 파일 구조와 인터페이스

아래는 원래 계획한 논리적 모듈 구분이다. 실제 구현은 `run.py`와 `configs/{smoke_05b,smoke_7b,pilot_05b,main_7b}.yaml`, `src/layerwise_lm/`에 있다. 실제 모듈명은 `model.py`, `data.py`, `metrics.py`, `evaluate.py`, `train.py`, `compare.py`, `cli.py`, `common.py`다. 저장은 Parquet 대신 압축 NPZ/CSV를 사용하고, 실제 CLI는 README를 참조한다.

```text
layerwise-lm-experiments/
  RESEARCH_GOALS.md
  EXPERIMENT_01_DESIGN.md
  pyproject.toml
  configs/{pilot_05b,before_7b,train_ce_7b,after_7b,compare_7b}.yaml
  src/layerwise_lm/
    model_adapter.py       # block/norm/head 접근, raw 출력 처리
    data.py                # 문서 복원, token shift, context/평가 mask
    observer.py            # hooks, chunk projection, compact 기록
    metrics.py             # hit/stable/coverage/CI
    evaluate.py            # teacher-forced 및 생성 모드
    train_lens.py          # 필수 B-CE, 선택 B-KL
    compare.py             # 고정 manifest의 학습 전후 paired 비교
    report.py              # CSV, PNG/PDF, Markdown 생성
  tests/
    test_alignment.py
    test_model_equivalence.py
    test_metrics.py
  runs/<comparison_id>/
    pair_manifest.json
    before_train/
    training/
    after_train/
    comparison.csv
    comparison.md
```

설정 예시:

```yaml
model_id: Qwen/Qwen2.5-0.5B-Instruct
model_revision: 7ae557604adf67be50417f59c2c2f167def9a775
mode: teacher_forced
readout: shared_logit_lens
dtype: bfloat16
seed: 42
allowed_physical_gpu_indices: [1, 2, 3]
preferred_physical_gpu: 1
layers: all  # embedding 제외, 1..L
dataset_id: Salesforce/wikitext
dataset_config: wikitext-2-raw-v1
split: validation  # 본 평가에서는 test
seq_len: 512
context_overlap: 64
batch_size: 1
projection_chunk_tokens: 32
max_eval_tokens: 8192
top_k: 5
use_cache: false
output_dir: runs/pilot_05b
```

실제 CLI는 `python run.py --config configs/pilot_05b.yaml --gpu 1`이다. 단계별 명령은 README에 정리한다. 확인한 라이브러리 버전은 `requirements-tested.txt`, model/dataset revision과 실행 환경은 run manifest에 저장한다.

## 10. 저장 스키마와 보고서

| 파일 | 핵심 내용 |
| --- | --- |
| `manifest.json` | config, model/data revision, 라이브러리·CUDA·GPU, 코드 commit 또는 source hash, mask/문서 선택 규약 |
| `targets.parquet` | sample/document ID, absolute target position, reference ID, teacher ID, run mode |
| `predictions.parquet` | target ID, layer, top-5 ID, reference logprob, max prob, entropy, margin |
| `token_metrics.parquet` | target별 D_ref, D_teacher, S_ref, S_teacher, no-hit, flip |
| `layer_metrics.csv` | layer별 정확도·일치율·top-5·NLL·CI |
| `summary.json` | 성공 subset 평균과 coverage, 실패 포함 oracle 비용, token/문서 수 |
| `examples.jsonl` | 일부 prefix와 layer별 token 변화; token ID와 사람이 읽는 문자열 병기 |
| `figures/*.{png,pdf}` | accuracy-depth, 최초 hit histogram/CDF, 안정 깊이, token×layer heatmap |
| `report.md` | 조건, 실제 결과, 해석과 한계 |
| `pair_manifest.json` | A/B의 모델·평가 manifest hash, 학습 조건과 checkpoint ID, 결과 경로 |
| `comparison.csv` / `comparison.md` | 학습 전·후 지표와 변화량, paired CI, hit 성공/실패 전이 |

기본 집계 단위는 token ID다. 문자열 공백 정리나 decode 후 비교로 정답을 판정하지 않는다. 긴 run은 shard 단위로 저장하고 완료 shard를 manifest에 기록하여 resume 때 중복 집계하지 않는다.

최종 보고서의 기본 표:

| 조건 | Final ref acc | Ref hit coverage | Mean first hit / hit only | Mean stable hit / eligible | No-hit | Teacher first-hit mean | Oracle block saving |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A: 학습 전 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 |
| B-CE: 학습 후 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 |
| 변화량 B−A | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 |

평균 최초 hit와 안정 깊이는 낮을수록, hit coverage와 oracle block saving은 높을수록 좋다. No-hit와 NLL은 낮을수록 좋다. 성공 subset이 바뀌는 지표의 차이는 공통-hit 분석과 함께 해석한다. 필수 A/B 표에 선택 B-KL 결과를 추가할 경우 별도 행으로 명시한다.

분포는 final-correct/final-incorrect, 문서, 문맥 길이, punctuation/whitespace/일반 token 등의 subset으로도 살펴본다. 드문 subset의 작은 표본을 과해석하지 않는다.

## 11. 구현 완료 판정과 검증

필수 검증은 수치 해석을 틀리게 할 수 있는 지점에 집중한다.

1. **Shift/mask 검증:** 짧은 token sequence로 position i가 x[i+1]을 예측하는지 확인한다. Padding·overlap target은 제외하고, 문서의 warm-up 이후 각 target이 chunk 경계를 포함해 정확히 한 번 집계되는지 검사한다.
2. **관측 무간섭:** observer 유무에 따라 원래 모델 최종 logits가 수치 허용 오차 내에서 같아야 한다.
3. **마지막 head 동등성:** raw last-block→final norm→LM Head가 원래 logits와 일치해야 한다. BF16 rounding 차이는 FP32 재검사와 margin으로 설명하고, double norm/index 오류는 허용하지 않는다.
4. **Causality/경로 검증:** 일부 prefix를 잘라 개별 forward한 결과가 같은 위치의 병렬 평가와 일치하는지 확인한다. 이후 생성 모드에서는 cache on/off도 비교한다.
5. **지표 검증:** 알려진 정답 패턴으로 최초·안정 깊이, hit 후 flip, no-hit, 실패 포함 비용을 손으로 계산한 값과 대조한다.
6. **불변 조건:** teacher agreement at L=100%, teacher D 존재율=100%, `F_ref(L)=coverage_ref`, `A_ref(L)=P(S_ref exists)`, CDF 단조 증가를 확인한다.
7. **학습 격리:** B 학습 전후 backbone/norm/head weight hash가 같고 adapter만 바뀌는지 확인한다. 학습 전 identity adapter가 A를 재현해야 한다.
8. **학습 전후 대응:** A/B의 평가 manifest, target ID, teacher token이 같아야 한다. Adapter checkpoint를 저장·복원한 후 예측이 재현되고, 학습된 parameter와 train/validation 사용 이력이 보고서에 남아야 한다.

이후 0.5B pilot에서 학습 전 평가·짧은 B-CE 학습·checkpoint 복원·학습 후 평가·비교 파일 생성을 모두 확인하고 VRAM/time을 기록한다. 이를 통과하면 7B에서 동일 흐름을 본 규모로 수행한다. E01은 **A 평가, B-CE 학습, 같은 test에서의 B-CE 평가, 학습 전후 비교 보고서가 모두 생성되었을 때만 완료**로 본다. 개선 여부는 실험 결과이며, 학습 후 개선이 없더라도 그 결과를 그대로 보고한다.

E01의 관측 hook 실행 시간은 instrumentation overhead를 포함한다. 원래 모델의 일반 inference 시간과 observer 시간은 구분해 기록한다. 실제 가속 판정은 후속 구현에서 hook 없이, prefill/decode 분리, warm-up 후 CUDA synchronization, 여러 번 반복한 wall-clock과 tokens/s로 수행한다.

## 12. 결과에 따른 다음 행동

- A/B-CE의 학습 전후 비교를 먼저 완료한다. B에서 이른 hit가 많고 안정 깊이도 얕다면 gate의 precision/coverage를 확인한다.
- A는 약하지만 B에서 개선되면 readout alignment가 병목이라는 가설을 지지한다. 새 토큰 중간 시작의 성공을 뜻하지는 않는다.
- A/B 모두 teacher 일치가 늦다면 backbone에 early-exit loss를 주는 E02를 검토한다. Frozen readout 실험만으로 공동 학습의 가능성을 기각하지 않는다.
- Reference no-hit가 높으면 최종 모델의 baseline 정확도와 자연어의 복수 정답 문제를 함께 확인한다. 같은 prefix의 동일 재실행은 해결책이 아니다.
- Teacher 일치는 일찍 달성되나 reference 정확도가 낮다면, 원래 모델의 행동 재현과 실제 정답 개선을 별도 목표로 선택한다.
- E01에서 후보가 보이면 시작 layer별 입력 adapter와 cache 규약을 먼저 구현한 E03으로 진행한다. 정상 경로의 hit 분포를 원형 경로에 그대로 대입해 절감률을 계산하지 않는다.

## 13. 참고 근거

- [CALM 논문](https://arxiv.org/abs/2207.07061): 토큰별 조기 종료, confidence와 누락된 과거 hidden representation 문제.
- [Tuned Lens 논문](https://arxiv.org/abs/2303.08112), [저자 코드](https://github.com/AlignmentResearch/tuned-lens): 고정 모델의 중간 표현에 affine readout을 학습하는 비교 조건.
- [LayerSkip 논문](https://arxiv.org/abs/2404.16710), [저자 코드](https://github.com/facebookresearch/LayerSkip): early-exit 학습과 self-speculative 검증의 비교 기준.
- [공식 Qwen2 소스](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py): block, final norm, LM Head와 KV cache 접근 확인. 실제 구현은 설치한 버전을 기준으로 다시 확인한다.
- [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext): E01 기본 영어 LM 데이터. 전처리·subset 선택은 이 문서에서 제안한 실험 규약이다.

하드웨어·캐시 revision·모델 크기는 로컬 확인값이며, 실험 규모·hyperparameter·파일 구조·후속 단계는 본 연구를 위한 설계 제안이다. 정답 도달 깊이와 성능 향상에 관한 측정값은 아직 없다.
