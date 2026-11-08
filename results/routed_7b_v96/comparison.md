# 실제 순환 경로 학습 전후 비교

정답을 맞히면 즉시 다음 token을 같은 block에 넣고, 거기서 계산을 계속한 실행 결과다.

| 지표 | 학습 전 | 학습 후 | 변화량 |
| --- | ---: | ---: | ---: |
| resolved_requested_fraction | 0.078125 | 0.0859375 | 0.0078125 |
| episode_completion_fraction | 0.0 | 0.0 | 0.0 |
| resolved_targets | 10 | 11 | 1 |
| failed_targets | 16 | 16 | 0 |
| unreached_targets | 102 | 101 | -1 |
| same_layer_token_injections | 10 | 11 | 1 |
| wrap_transitions | 59 | 59 | 0 |
| mean_visits_per_hit | 29.5 | 27.0 | -2.5 |
| mean_exit_layer_given_hit | 26.4 | 23.90909090909091 | -2.4909090909090885 |
| decode_visits_per_resolved_target_including_failures | 183.1 | 166.63636363636363 | -16.46363636363637 |
| mean_nll_per_visited_block | 14.13101467641924 | 10.930351879285618 | -3.200662797133621 |
| decode_block_visits | 1831 | 1833 | 2 |

공통 성공 target: 10, 공통 target의 실제 block 방문 수 변화: -2.6.

평가는 실제 다음 token label로 분기를 선택하는 oracle 실험이다. 실제 배포용 exit gate는 포함하지 않는다.

예산을 소진하면 해당 episode를 종료한다. 정답을 강제로 주입하지 않으며 이후 target은 unreached로 남긴다.

성공률·완료율과 함께 계산량을 읽어야 한다. 일찍 실패한 실행의 작은 총 방문 수를 가속으로 해석하지 않는다.

Trace에는 매 block의 input/target/layer/action/next_layer/injected token을 저장한다. KV는 방문한 위치만 가지며 재방문은 같은 위치를 교체한다.
