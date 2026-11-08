# 실제 순환 경로 학습 전후 비교

정답을 맞히면 즉시 다음 token을 같은 block에 넣고, 거기서 계산을 계속한 실행 결과다.

| 지표 | 학습 전 | 학습 후 | 변화량 |
| --- | ---: | ---: | ---: |
| resolved_requested_fraction | 0.0390625 | 0.0546875 | 0.015625 |
| episode_completion_fraction | 0.0 | 0.0 | 0.0 |
| resolved_targets | 5 | 7 | 2 |
| failed_targets | 16 | 16 | 0 |
| unreached_targets | 107 | 105 | -2 |
| same_layer_token_injections | 5 | 7 | 2 |
| wrap_transitions | 21 | 23 | 2 |
| mean_visits_per_hit | 21.6 | 22.428571428571427 | 0.8285714285714256 |
| mean_exit_layer_given_hit | 21.6 | 20.571428571428573 | -1.0285714285714285 |
| decode_visits_per_resolved_target_including_failures | 175.2 | 132.14285714285714 | -43.05714285714285 |
| mean_nll_per_visited_block | 13.548286093560527 | 13.03176474748431 | -0.5165213460762175 |
| decode_block_visits | 876 | 925 | 49 |

공통 성공 target: 5, 공통 target의 실제 block 방문 수 변화: -0.2.

평가는 실제 다음 token label로 분기를 선택하는 oracle 실험이다. 실제 배포용 exit gate는 포함하지 않는다.

예산을 소진하면 해당 episode를 종료한다. 정답을 강제로 주입하지 않으며 이후 target은 unreached로 남긴다.

성공률·완료율과 함께 계산량을 읽어야 한다. 일찍 실패한 실행의 작은 총 방문 수를 가속으로 해석하지 않는다.

Trace에는 매 block의 input/target/layer/action/next_layer/injected token을 저장한다. KV는 방문한 위치만 가지며 재방문은 같은 위치를 교체한다.
