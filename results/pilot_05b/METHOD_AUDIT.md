# 0.5B pilot: 실제 학습 목표와 평가 데이터 구성

확인일: 2026-11-02 (Asia/Seoul). 실행 코드, 저장된 JSONL 및 manifest를 읽어 확인한 기록이다. 기존 실행 코드·checkpoint·결과는 변경하지 않았다.

## 무엇을 학습했는가

Qwen2.5-0.5B-Instruct의 Transformer/embedding/final norm/기존 LM Head는 고정했다. 1~23번째 block의 raw hidden state를 공유 LM Head의 입력 공간으로 보정하는 layer별 dense residual adapter만 학습했다.

```text
입력: 문서의 token prefix x[0], ..., x[t]
정답: 원문에 실제 기록된 다음 token x[t+1]

h_l: 고정된 backbone의 block l 출력
h'_l = h_l + A_l h_l + b_l
z_l = 기존 LM Head(기존 final norm(h'_l))

loss = 1/23 * sum_{l=1..23} CE(z_l, x[t+1])
```

학습 전은 모든 A_l=0, b_l=0인 identity 상태다. 학습 후는 같은 구조의 adapter parameter만 업데이트한 상태다. Dense adapter는 hidden 896×896 weight와 bias를 각 중간 layer에 가지며, 총 trainable parameter는 18,485,376개다. 최종 24번째 layer에는 adapter를 학습하지 않는다.

일반적인 next-token supervised learning을 중간 readout에 적용했다. 정답 token으로 CE loss를 계산하며, 최종 layer 분포를 정답으로 삼는 KL distillation은 이번 실행에 사용하지 않았다. Adapter 출력은 다음 Transformer block의 입력으로 전달하지 않는다.

따라서 현재 결과가 보여주는 것은 '고정된 중간 표현에서 올바른 다음 token을 더 잘 읽어낼 수 있다'는 것이다. Backbone을 early-exit용으로 공동 학습하거나, token 시작 layer·순환 경로·exit gate를 학습한 결과는 아니다.

## 실제 데이터 분리

Dataset: `Salesforce/wikitext`, `wikitext-2-raw-v1`. Revision: `b08601e04326c79dfdd32d625aee71d232d685c3`.

| 용도 | 원본 split | 선택된 문서 수 | Target tokens | 입력 chunk 수 |
| --- | --- | ---: | ---: | ---: |
| Adapter 학습 | 공식 train | 12 | 32,768 | 80 |
| Checkpoint 선택 | 공식 validation의 문서 hash 짝수 그룹 | 1 | 1,024 | 3 |
| 학습 전후 평가 | 공식 validation의 문서 hash 홀수 그룹 | 5 | 8,192 | 21 |

**0.5B pilot은 공식 test split을 평가에 사용하지 않았다.** 별도 평가 subset은 공식 validation의 checkpoint 선택 그룹과 다른 문서에서 만들었다. 7B 본 실험 설정은 공식 test split을 사용한다.

문서는 최상위 article 제목으로 복원했다. 공식 validation 문서 ID hash의 짝·홀로 checkpoint 선택/평가 pool을 나눈 후, 각 pool의 문서 순서를 seed 42로 섞었다. 문서 안의 token 순서는 유지하고 순서대로 chunk를 만들다가 target cap에 도달하면 마지막 문서 중간에서 잘랐다. 전체 pool의 모든 문서를 골고루 쓰는 방식은 아니다.

저장된 문서의 content-hash prefix 기준 train/validation/evaluation 간 겹침은 없고, 평가의 `(document_id, absolute_target_position)` 중복도 없다. 이 점은 Qwen의 기존 사전학습 데이터에 WikiText가 포함되지 않았다는 보장을 뜻하지 않는다.

Checkpoint 선택용 문서: **Thelnetham Windmill**, 1,024 target tokens.

평가 문서별 실제 target 수:

| 문서 | Target tokens |
| --- | ---: |
| Exit Through the Kwik-E-Mart | 2,095 |
| Hurricane Beatriz (2011) | 1,513 |
| 2011–12 Michigan Wolverines men's basketball team | 2,923 |
| Tropical Storm Josephine (2008) | 1,411 |
| U.S. Route 2 in Michigan | 250 |
| 합계 | 8,192 |

긴 문서가 더 많은 평가 token을 제공하므로 문서당 가중치가 같은 표본은 아니다. 특히 한 문서가 2,923 tokens를 차지한다. 8,192개의 독립적인 문장이나 질문이 아니라, 5개 문서에서 선택한 다음-token 예측 위치 8,192개다.

## 입력과 정답을 만드는 방법

- 문서별 독립적으로 token화하며, 서로 다른 문서를 이어 붙이지 않는다. Chat template 및 인위적인 BOS/EOS는 추가하지 않는다.
- 최대 512-token chunk, 64-token overlap. Local target index j가 64 이상인 위치만 평가한다.
- Target token `input_ids[j]`를 예측하는 hidden state는 `hidden[j-1]`이다. Causal attention으로 해당 hidden은 target과 그 이후 token을 볼 수 없다.
- 예: chunk 0..511은 target 64..511, 다음 chunk 448..959는 target 512..959를 평가한다. 앞의 64 tokens는 context로 쓰고 overlap target은 두 번 세지 않는다.
- Special-token target은 제외한다. Target 수는 context로만 사용한 token 수를 포함하지 않는다.

실제 첫 평가 target: local target index 64, hidden index 63. Prefix 끝부분은 `… in the United States`, 정답 다음 token은 공백을 포함한 `" on"`(token ID 389)이다. 각 layer의 top-1 token ID가 389인지 비교한다.

평가는 teacher forcing이다. 각 위치에서 앞 문맥은 원문 token이며, 앞서 모델이 틀렸다고 해도 그 오답을 다음 위치의 입력으로 누적하지 않는다. Causal mask가 적용된 forward로 여러 위치를 병렬 평가한다. 자유 생성이나 문장 단위 의미적 정답 평가와 구분해야 한다.

## 학습과 비교 규칙

- 위 학습 subset을 1 epoch, AdamW LR 1e-4, 15 optimizer updates.
- Validation의 평균 중간-layer NLL이 가장 낮은 학습 checkpoint를 선택. 이번 실행에서는 step 15가 선택됨.
- A/B는 동일한 8,192 target IDs, prefix, chunk 길이·위치, target mask를 사용. Evaluation 중 parameter 업데이트 없음.
- 주 정답 기준은 원문의 다음 token ID와의 exact match. 최종 layer 예측과의 teacher 일치도는 별도 지표.
- 최초 hit는 24개 layer 중 reference 정답이 처음 나타난 깊이. 중간에 한 번 맞았다가 이후 틀린 경우도 최초 hit에 포함. 안정 깊이와 미도달 비율을 별도로 보고함.

## 해석 범위

이 결과는 frozen-backbone readout 학습의 작은 가능성 확인 실험이다. Checkpoint 선택 문서가 1개, 평가 문서가 5개이고 seed도 1개다. 원래 모델의 추론 능력 향상, 실제 decoding 가속, 학습된 exit 판정, 중간 layer에서 새 token 시작 또는 L→1 재순환의 성공을 입증하지 않는다.

관련 파일: [설정](../../configs/pilot_05b.yaml), [데이터 manifest](data/manifest.json), [학습 기록](training/done.json), [결과 보고서](comparison.md).
