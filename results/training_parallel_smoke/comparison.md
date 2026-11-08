# 새 iteration 학습 전후 비교

정답 정렬: next_token; iteration: global.

| 지표 | 학습 전 | 학습 후 | 변화량 |
| --- | ---: | ---: | ---: |
| processed_targets | 6 | 6 | 0 |
| unforced_hits | 1 | 1 | 0 |
| forced_targets | 5 | 5 | 0 |
| unforced_hit_fraction | 0.16666666666666666 | 0.16666666666666666 | 0.0 |
| all_correct_without_forcing_fraction | 0.0 | 0.0 | 0.0 |
| visited_blocks | 145 | 145 | 0 |
| mean_ce_per_visited_block | 14.969239909073401 | 14.939997261968152 | -0.029242647105249375 |

실패한 token은 loss를 남긴 뒤 정답 입력으로 다음 token 학습을 계속한다. 따라서 processed_targets는 생성 성공 수가 아니다.

unforced_hits만 head가 정답을 직접 맞힌 token이며 forced_targets는 iteration 예산 후 정답을 공급한 token이다.

Token index는 context를 제외한 target sequence의 0-based index다. 강제 전환은 layer 1부터, 정답 전환은 같은 layer부터 시작한다.

input_token 정렬은 이미 입력에 있는 token 복원 과제이며 next_token 예측 성능과 구분한다.
