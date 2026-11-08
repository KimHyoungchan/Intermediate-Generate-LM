# 새 iteration 학습 전후 비교

정답 정렬: next_token; iteration: global.

| 지표 | 학습 전 | 학습 후 | 변화량 |
| --- | ---: | ---: | ---: |
| processed_targets | 8192 | 8192 | 0 |
| unforced_hits | 651 | 2537 | 1886 |
| forced_targets | 7541 | 5655 | -1886 |
| unforced_hit_fraction | 0.0794677734375 | 0.3096923828125 | 0.230224609375 |
| all_correct_without_forcing_fraction | 0.0 | 0.001953125 | 0.001953125 |
| visited_blocks | 197146 | 187981 | -9165 |
| mean_ce_per_visited_block | 14.824491913256573 | 7.9136246132112635 | -6.910867300045309 |

실패한 token은 loss를 남긴 뒤 정답 입력으로 다음 token 학습을 계속한다. 따라서 processed_targets는 생성 성공 수가 아니다.

unforced_hits만 head가 정답을 직접 맞힌 token이며 forced_targets는 iteration 예산 후 정답을 공급한 token이다.

Token index는 context를 제외한 target sequence의 0-based index다. 강제 전환은 layer 1부터, 정답 전환은 같은 layer부터 시작한다.

input_token 정렬은 이미 입력에 있는 token 복원 과제이며 next_token 예측 성능과 구분한다.
