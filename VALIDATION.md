# 구현 검증 기록

검증 시각과 사용 장치·라이브러리 세부 정보는 각 run의 JSON에 기록된다.

- CPU 단위 검사: 6개 통과.
- 0.5B/7B 실제 모델의 학습 전 평가 → adapter 학습 → checkpoint 복원 → 학습 후 평가 → 비교표·그래프 생성: 모두 통과.
- 전체 7B 1M-token 학습 및 100k-token test 평가는 아직 수행하지 않음.
- `--resume`: 완료된 데이터·평가·학습 단계를 checksum 검증 후 재사용하고 비교 보고서를 재생성하는 흐름 통과.

| 항목 | 0.5B smoke | 7B smoke |
| --- | --- | --- |
| 사용 물리 GPU | 1 | 2 |
| Train / validation / evaluation tokens | 256 / 64 / 128 | 64 / 16 / 32 |
| 관측 layer 수 | 24 | 28 |
| 학습 구조 | Dense readout | Rank-64 readout |
| Optimizer updates | 3 | 1 |
| Validation 중간-layer NLL | 12.020916 → 10.747119 | 11.367800 → 11.296964 |
| 학습 중 peak allocated CUDA memory | 1.295 GiB | 14.454 GiB |
| Backbone/norm/head weight hash 불변 | 통과 | 통과 |
| 학습된 checkpoint가 초기 상태와 다름 | 통과 | 통과 |
| Raw 마지막 block → norm/head의 logits 동등성 | 최대 오차 0 | 최대 오차 0 |
| 같은 길이에서 미래 suffix 변경 시 과거 logits 불변 | 최대 오차 0 | 최대 오차 0 |
| A/B target·teacher 및 최종 layer 예측 동일 | 통과 | 통과 |

위 NLL과 깊이 변화는 pipeline 검증용이며, 성능 개선의 근거로 해석하지 않는다. 각 smoke의 평가 문서는 1개뿐이므로 bootstrap CI도 통계적 의미가 없다. 7B의 한 번 업데이트는 일부 top-1/깊이를 바꾸지 않았으며 결과를 그대로 저장했다.

- [0.5B smoke 보고서](results/smoke_05b/comparison.md)
- [7B smoke 보고서](results/smoke_7b/comparison.md)

본 실험의 고정 데이터도 준비했다. Train 1,000,000 tokens / 261 documents, validation 8,192 tokens / 2 documents, test 100,000 tokens / 21 documents다. Validation이 2개 문서에 집중되어 있으므로 본격적인 일반화 평가에서는 더 넓은 validation subset으로 별도 재현할 수 있다. 이 변경은 새 config/output_dir에서 수행한다. Dataset revision과 파일 checksum은 `runs/main_7b/data/manifest.json`에 있다.

BF16에서 입력 길이를 바꾼 prefix 실행과 전체 실행 사이에는 최대 약 0.45–0.48의 vocabulary logit 차이가 관측되었다. 동일 길이의 미래 token 치환 검사에서는 오차가 0이었다. 길이에 따른 수치 차이는 summary의 진단 항목에 보존하며, A/B는 같은 prefix·길이·projection chunk로 비교한다.
