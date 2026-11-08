# 학습 전후 비교

Model: `Qwen/Qwen2.5-7B-Instruct`

평가 token: 100,000, 문서: 21. 동일 prefix·mask의 paired 비교.

| 지표 | 학습 전 | 학습 후 | 변화량 B−A | 변화량 95% CI |
| --- | ---: | ---: | ---: | --- |
| reference_hit_coverage | 0.632190 | 0.696700 | 0.064510 | [0.058507, 0.069691] |
| reference_no_hit_rate | 0.367810 | 0.303300 | -0.064510 | [-0.069691, -0.058507] |
| reference_first_hit_mean | 24.472500 | 14.033300 | -10.439201 | [-11.193776, -9.658454] |
| reference_stable_mean | 25.766776 | 19.831200 | -5.935576 | [-6.533627, -5.348563] |
| teacher_first_hit_mean | 25.122930 | 15.872060 | -9.250870 | [-9.827916, -8.650027] |
| oracle_fullscan_mean | 25.769950 | 18.269400 | -7.500550 | [-8.001914, -6.955766] |
| common_hit_first_depth | 24.474267 | 14.628991 | -9.845276 | [-10.656831, -9.031478] |

실패→성공: 8,251; 성공→실패: 1,800; 공통 성공: 61,419; 둘 다 실패: 28,530.

Adapter optimizer steps: 444; selected step: 444.

Validation NLL: 11.788287 → 5.345070.

Final-layer accuracy: 0.559420 → 0.559420 (고정 backbone/head).

최초·안정 깊이와 NLL은 낮을수록 좋고 coverage는 높을수록 좋다. 성공 subset 평균의 변화는 공통 성공 분석과 함께 해석한다.

Oracle block 수는 LM Head·gate·KV 보완 비용을 제외한다. 실제 속도 향상이나 순환 경로의 성공을 나타내지 않는다.

문서 단위 bootstrap CI다. 문서 수가 적은 smoke/pilot은 성능 결론에 사용하지 않는다.

![Layer accuracy](figures/accuracy.png)

![First-hit CDF](figures/first_hit_cdf.png)
