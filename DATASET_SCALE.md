# 전체 데이터와 실제 학습량 점검

2026-11-02. 현재 `training_revision_05b`와 `training_revision_7b` 실행은 **동작 확인용 소규모 실험**이다. 전체 WikiText-2를 학습한 실행이 아니며, 이 결과만으로 제안한 구조의 학습 가능성을 판단할 수 없다. 기존의 1 epoch는 아래에서 추출한 train 128개 episode에 대한 1 epoch다.

## 데이터의 정체

- 데이터: `Salesforce/wikitext`, 구성 `wikitext-2-raw-v1`.
- 고정 revision: `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Wikipedia 문서에서 구성한 영어 언어 모델링 데이터다. 질문·답변 쌍이나 instruction 데이터가 아니다. 본 실험의 정답은 원문에서 다음에 나타나는 token이다.
- 공식 train/validation/test 분할을 사용한다. 하나의 작은 표본을 다시 세 분할한 것이 아니라, 각 공식 split에서 별도로 작은 표본을 추출했다.
- 공식 설명과 원본 행 수: https://huggingface.co/datasets/Salesforce/wikitext#data-splits

## 전체 데이터 규모와 실제 사용량

| Split | 원본 text 행 수 | 전체 Qwen 토큰 수 | 실제 사용 episode | 실제 정답 token 수 |
| --- | ---: | ---: | ---: | ---: |
| Train | 36,718 | 2,517,232 | 128 | 1,024 |
| Validation | 3,760 | 262,210 | 8 | 64 |
| Test | 4,358 | 298,938 | 16 | 128 |
| 합계 | 44,836 | 3,078,380 | 152 | 1,216 |

원본 행에는 빈 줄과 제목·문단이 포함되므로 행 수를 문서 수로 해석하지 않는다. 실험의 제목 기준 문서 복원 코드를 적용하면 train 629, validation 60, test 62개 단위가 나온다. 이는 우리 전처리 코드가 복원한 단위 수다.

토큰 수는 실험과 동일한 문서 복원 후 `add_special_tokens=False`로 전체 문서를 직접 토큰화하여 집계했다. 제목과 개행을 포함하며, 문서별 최초 context를 제외하기 전의 총량이다. 실제 사용한 0.5B와 7B tokenizer로 각각 집계했고 이 데이터에서는 수가 동일했다. 전체 token 수를 그대로 학습 target 수로 간주하면 안 된다.

- 0.5B tokenizer revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
- 7B tokenizer revision: `a09a35458c702b33eeacc393d103063234e8bc28`.
- 기계 판독용 실측: [DATASET_SCALE_AUDIT.json](DATASET_SCALE_AUDIT.json).

## 실제 작은 표본의 의미

한 episode는 context 32 tokens + 정답 8 tokens다. Train은 128 × 8 = 1,024개의 서로 다른 target 위치이며, context 4,096개를 더해도 입력 창 총량은 5,120 tokens에 그친다. Context token은 해당 episode의 직접적인 CE 정답 수에 포함하지 않는다. 각 split 안에서 target 위치가 중복되지 않도록 표본을 뽑았다.

실제 학습한 1,024 targets는 전체 train 2,517,232 tokens 대비 약 **0.0407%**다. 같은 target에서 여러 layer를 방문하여 CE를 계산한 횟수는 새로운 원문 token을 학습한 수가 아니다. 특히 0.5B의 24,653회 block 방문을 24,653개의 서로 다른 정답 token으로 해석하면 안 된다.

두 모델 모두 1 epoch, 64 optimizer updates였다. 원본 weight를 고정하고 LoRA와 token 주입·순환·출력 보정 parameter를 학습했다. 0.5B 학습 단계의 기록 시간은 약 88.59초다. 빠르게 끝난 주요 이유는 이처럼 작은 데이터 설정이다.

따라서 현재 결과는 경로 전환·loss·gradient·checkpoint 저장의 동작 확인 결과로 제한한다. 본 실험에는 더 큰 학습량과 validation/test 표본이 필요하며, 전체 train을 사용하는 학습은 아직 실행하지 않았다. 기존 test를 반복 확인했으므로 앞으로의 모델·설정 선택에는 validation을 사용하고 최종 test 해석에는 그 노출 이력을 함께 기록한다.

실제 표본과 실행 근거:

- [0.5B 데이터 manifest](results/training_revision_05b/data/manifest.json)
- [0.5B 학습 완료 기록](results/training_revision_05b/training/done.json)
- [7B 데이터 manifest](results/training_revision_7b/data/manifest.json)
- [7B 학습 완료 기록](results/training_revision_7b/training/done.json)
