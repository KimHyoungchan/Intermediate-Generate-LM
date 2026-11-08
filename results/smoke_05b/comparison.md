# 학습 전후 비교

Model: `Qwen/Qwen2.5-0.5B-Instruct`

평가 token: 128, 문서: 1. 동일 prefix·mask의 paired 비교.

| 지표 | 학습 전 | 학습 후 | 변화량 B−A | 변화량 95% CI |
| --- | ---: | ---: | ---: | --- |
| reference_hit_coverage | 0.515625 | 0.500000 | -0.015625 | [-0.015625, -0.015625] |
| reference_no_hit_rate | 0.484375 | 0.500000 | 0.015625 | [0.015625, 0.015625] |
| reference_first_hit_mean | 21.257576 | 20.281250 | -0.976326 | [-0.976326, -0.976326] |
| reference_stable_mean | 21.896552 | 21.120690 | -0.775862 | [-0.775862, -0.775862] |
| teacher_first_hit_mean | 21.851562 | 20.820312 | -1.031250 | [-1.031250, -1.031250] |
| oracle_fullscan_mean | 22.585938 | 22.140625 | -0.445312 | [-0.445312, -0.445312] |
| common_hit_first_depth | 21.206349 | 20.238095 | -0.968254 | [-0.968254, -0.968254] |

실패→성공: 1; 성공→실패: 3; 공통 성공: 63; 둘 다 실패: 61.

Adapter optimizer steps: 3; selected step: 3.

Validation NLL: 12.020916 → 10.747119.

Final-layer accuracy: 0.453125 → 0.453125 (고정 backbone/head).

최초·안정 깊이와 NLL은 낮을수록 좋고 coverage는 높을수록 좋다. 성공 subset 평균의 변화는 공통 성공 분석과 함께 해석한다.

Oracle block 수는 LM Head·gate·KV 보완 비용을 제외한다. 실제 속도 향상이나 순환 경로의 성공을 나타내지 않는다.

문서 단위 bootstrap CI다. 문서 수가 적은 smoke/pilot은 성능 결론에 사용하지 않는다.

![Layer accuracy](figures/accuracy.png)

![First-hit CDF](figures/first_hit_cdf.png)
