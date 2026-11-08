# 학습 전후 비교

Model: `Qwen/Qwen2.5-7B-Instruct`

평가 token: 32, 문서: 1. 동일 prefix·mask의 paired 비교.

| 지표 | 학습 전 | 학습 후 | 변화량 B−A | 변화량 95% CI |
| --- | ---: | ---: | ---: | --- |
| reference_hit_coverage | 0.875000 | 0.875000 | 0.000000 | [0.000000, 0.000000] |
| reference_no_hit_rate | 0.125000 | 0.125000 | 0.000000 | [0.000000, 0.000000] |
| reference_first_hit_mean | 24.928571 | 24.928571 | 0.000000 | [0.000000, 0.000000] |
| reference_stable_mean | 25.192308 | 25.192308 | 0.000000 | [0.000000, 0.000000] |
| teacher_first_hit_mean | 25.156250 | 25.156250 | 0.000000 | [0.000000, 0.000000] |
| oracle_fullscan_mean | 25.312500 | 25.312500 | 0.000000 | [0.000000, 0.000000] |
| common_hit_first_depth | 24.928571 | 24.928571 | 0.000000 | [0.000000, 0.000000] |

실패→성공: 0; 성공→실패: 0; 공통 성공: 28; 둘 다 실패: 4.

Adapter optimizer steps: 1; selected step: 1.

Validation NLL: 11.367800 → 11.296964.

Final-layer accuracy: 0.812500 → 0.812500 (고정 backbone/head).

최초·안정 깊이와 NLL은 낮을수록 좋고 coverage는 높을수록 좋다. 성공 subset 평균의 변화는 공통 성공 분석과 함께 해석한다.

Oracle block 수는 LM Head·gate·KV 보완 비용을 제외한다. 실제 속도 향상이나 순환 경로의 성공을 나타내지 않는다.

문서 단위 bootstrap CI다. 문서 수가 적은 smoke/pilot은 성능 결론에 사용하지 않는다.

![Layer accuracy](figures/accuracy.png)

![First-hit CDF](figures/first_hit_cdf.png)
