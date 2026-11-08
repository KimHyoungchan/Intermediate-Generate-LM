# 실제 순환 경로 학습 전후 비교

정답을 맞히면 즉시 다음 token을 같은 block에 넣고, 거기서 계산을 계속한 실행 결과다.

| 지표 | 학습 전 | 학습 후 | 변화량 |
| --- | ---: | ---: | ---: |
| resolved_requested_fraction | 0.0390625 | 0.0390625 | 0.0 |
| episode_completion_fraction | 0.0 | 0.0 | 0.0 |
| resolved_targets | 5 | 5 | 0 |
| failed_targets | 16 | 16 | 0 |
| unreached_targets | 107 | 107 | 0 |
| same_layer_token_injections | 5 | 5 | 0 |
| wrap_transitions | 53 | 53 | 0 |
| mean_visits_per_hit | 21.6 | 21.2 | -0.40000000000000213 |
| mean_exit_layer_given_hit | 21.6 | 21.2 | -0.40000000000000213 |
| decode_visits_per_resolved_target_including_failures | 328.8 | 328.4 | -0.4000000000000341 |
| mean_nll_per_visited_block | 14.076384179566029 | 12.546956280207809 | -1.52942789935822 |
| decode_block_visits | 1644 | 1642 | -2 |

공통 성공 target: 5, 공통 target의 실제 block 방문 수 변화: -0.4.

평가는 실제 다음 token label로 분기를 선택하는 oracle 실험이다. 실제 배포용 exit gate는 포함하지 않는다.

예산을 소진하면 해당 episode를 종료한다. 정답을 강제로 주입하지 않으며 이후 target은 unreached로 남긴다.

성공률·완료율과 함께 계산량을 읽어야 한다. 일찍 실패한 실행의 작은 총 방문 수를 가속으로 해석하지 않는다.

Trace에는 매 block의 input/target/layer/action/next_layer/injected token을 저장한다. KV는 방문한 위치만 가지며 재방문은 같은 위치를 교체한다.
