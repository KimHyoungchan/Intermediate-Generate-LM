# 실제 순환 경로 구현 검증

확인일: 2026-11-02 (Asia/Seoul).

## 사용자 실행: 96회 예산 결과

사용자가 `routed_05b_v96.yaml`, `routed_7b_v96.yaml`의 학습 전후 전체 흐름을 실행 완료했다. 아래의 이전 48/56회 기록과 별개다.

7B 96회 실행은 학습 전 평가 5.31초, 학습 단계 133.35초, 학습 후 평가 5.30초였다. 모델 로딩 등 단계 외 시간은 별도다. Train은 128 episodes × 8 requested targets = 1,024 slots, 1 epoch의 작은 pilot 설정이다.

실제 7B 학습 trace를 집계하면 accepted 108개 + budget-exhausted 128개 = **236개의 고유 target 위치에 도달**했고, **788개는 미도달**이었다. 모든 episode가 96회 예산 안에 현재 target을 맞히지 못한 지점에서 종료됐다. CE를 계산한 block 방문은 15,338회, optimizer update는 476회, 선택된 checkpoint는 update 448이다. Update 하나의 기준은 약 32회의 block 방문 누적이며 과거 readout 실험의 2,048 target-token 누적과 단위가 다르다.

학습 parameter는 6,096,412개이며 Transformer LoRA·entry/memory·wrap·exit군이 모두 바뀌었다. 기존 base weight hash는 불변이었다. 따라서 실제 학습은 수행됐으며, 빠른 시간은 작은 데이터와 중도 종료에 따른 실제 학습량 감소에 크게 기인한다. 7B가 충분히 학습됐다는 결과로 해석하지 않는다.

7B test는 학습 전 10/128 → 학습 후 11/128 targets를 처리했으며, 완주 episode는 양쪽 모두 0/16이다. 결과: [7B 96회 보고서](results/routed_7b_v96/comparison.md), [7B 학습 기록](results/routed_7b_v96/training/done.json). 0.5B 96회 결과: [보고서](results/routed_05b_v96/comparison.md).

## 동작 검증

- 경로·KV·gradient CPU 검사 6개 통과.
- 실제 Qwen2.5-0.5B의 최소 실행과 128-episode 학습 전후 pilot 완료, 물리 GPU 1 사용.
- 성공 시 같은 layer에서 새 token으로 다음 예측을 시작함.
- 실패 시 같은 token의 상태로 다음 layer 및 L→1 순환을 실행함.
- 예산 소진 시 정답을 강제로 주입하지 않고 episode를 종료함.
- 실제 trace 전체에서 위 전환 규칙을 검사해 통과함.
- Transformer q/v LoRA, token entry/memory, wrap, exit parameter가 모두 실제로 바뀜.
- 기존 base weight는 hash 검사에서 불변.

실제 0.5B smoke trace의 한 전환:

```text
target 0 / input position 7 / layer 20
  predicted token ID 304 == target ID 304 → emit

target 1 / input position 8 / layer 20
  input token ID 304로 다음 예측 시작
```

첫 token의 layer 21~24는 실행하지 않았다. Synthetic 분기 예시가 아니라 실제 모델 실행에서 기록된 전환이다.

## 수정된 0.5B pilot 결과

학습: 공식 train의 128개 문서에서 128 episodes, 각 context 32 / continuation 8 tokens, 1 epoch. Nominal target slot은 1,024개지만 실패 시 episode를 종료하므로 실제로 도달한 학습 target은 **192개(accept 64 + budget 실패 128)**였고, **832개는 미도달**이다. 실제 CE를 계산한 block 방문 수는 7,482회, optimizer update는 228회다. Validation으로 선택된 checkpoint는 update 224다.

Validation: 공식 validation의 8문서·8 episodes·64 requested targets. Test: 공식 test의 16문서·16 episodes·128 requested targets. Token당 방문 예산은 48 blocks다.

| 지표 | 학습 전 | 학습 후 |
| --- | ---: | ---: |
| 요청 target | 128 | 128 |
| 맞히고 실제 다음 입력으로 진행한 target | 5 | 7 |
| 예산 실패 target | 16 | 16 |
| 실패로 인해 도달하지 못한 target | 107 | 105 |
| 완주한 episode | 0/16 | 0/16 |
| 실제 same-layer token 주입 | 5 | 7 |
| L→1 wrap | 21 | 23 |
| Decode block 방문 | 876 | 925 |

학습 후의 총 방문 수가 더 많은 것은 더 많은 target을 진행했기 때문이다. 공통 성공 target 5개에서 평균 방문 수 차이는 −0.2 block이다. 전체적으로 이 작은 설정에서 안정적으로 문장을 이어간다고 결론낼 수 없다. 기존 readout 실험의 21.30→14.94 결과를 이 표와 혼합하지 않는다.

학습 전후·학습 trace에서 검사한 실제 전환 수:

| Trace | Same-layer emit | Wrap | Budget 종료 |
| --- | ---: | ---: | ---: |
| 학습 전 test | 5 | 21 | 16 |
| 학습 중 | 64 | 192 | 128 |
| 학습 후 test | 7 | 23 | 16 |

선택된 checkpoint의 parameter 변화 L2: Transformer LoRA 2.8543, token entry/memory 0.1484, wrap 0.4633, exit scale 0.3809. 따라서 중간 출력 모듈만 업데이트한 기존 실험과 학습 범위가 다르다.

- [0.5B 실제 경로 보고서](results/routed_05b/comparison.md)
- 학습 전 상세 경로: 원본 로컬 `runs/routed_05b/before_train/trace.jsonl`에 보존. 공개 snapshot에서는 대용량 trace를 제외했다.
- 학습 후 상세 경로: 원본 로컬 `runs/routed_05b/after_train/trace.jsonl`에 보존. 공개 snapshot에서는 대용량 trace를 제외했다.
- [학습 기록](results/routed_05b/training/done.json)
- [실행 명령과 정확한 규칙](ROUTED_EXPERIMENT.md)

## 7B

별도 물리 GPU 3에서 최소 학습 전후 실행과 checkpoint 복원·비교 보고서 생성까지 완료했다. Train 4 episodes × 2 requested targets, validation/test 각 2 episodes × 2 requested targets, 14 updates다. Test는 학습 전후 모두 1/4 target을 해결했고 same-layer 주입 1회, wrap 3회를 기록했다. 세 trace의 전환 검사도 통과했다.

이 작은 학습 표본에서는 첫 target을 맞힌 episode가 없어 학습 중 token-entry 모듈은 업데이트되지 않았다. Transformer LoRA·wrap·exit은 업데이트됐다. 7B test의 실제 동일-layer 재진입 경로는 실행됐으며, entry/memory의 실제 학습은 위 0.5B 실행과 tiny Transformer gradient 검사에서 확인했다. 이 구분을 성공 주장에 섞지 않는다.

7B 최소 학습의 peak allocated CUDA memory는 약 14.23 GiB다. [7B 최소 실행 보고서](results/routed_smoke_7b/comparison.md). 본 규모 `routed_7b.yaml`은 아직 실행하지 않았다. 기존 GPU 2의 readout 학습과 새 경로 검증은 별도 작업이다.
