# Intermediate-Generate-LM

중간 레이어에서 다음 토큰을 예측하고, 정답을 맞힌 레이어에서 다음 토큰 계산을 이어가며, 마지막 레이어 이후에는 상태를 첫 레이어로 전달하는 LLM 실험 코드다. Qwen2.5-0.5B-Instruct와 7B-Instruct를 사용한다.

이 저장소는 실험 코드, 설정, CPU 검사, 핵심 결과만 포함한다. 원본 데이터·모델 weight·학습 checkpoint·매 block trace는 로컬에 보존하고 Git에 포함하지 않는다. 공개 결과는 `results/`, 재실행 출력은 Git에서 제외한 `runs/`에 저장한다.

## 현재 확인된 결과

0.5B 확대 실험은 train 131,072 samples × 3 epochs를 완료했다. Sample은 context 32 + target 8 tokens이며, 서로 다른 train target 위치는 1,048,576개, 3 epochs 누적 처리량은 3,145,728개다. 물리 GPU 1·2·3에서 학습했고 validation으로 선택한 checkpoint를 같은 test 1,024 samples·8,192 targets에서 평가했다.

| 지표 | 학습 전 | 학습 후 |
| --- | ---: | ---: |
| 제한 안에 직접 맞힌 target | 651/8,192 (7.95%) | 2,537/8,192 (30.97%) |
| 정답 강제 공급 없이 8 targets 모두 성공한 sample | 0/1,024 | 2/1,024 (0.195%) |
| 성공한 target의 평균 실제 block 방문 | 21.31 | 5.62 |
| 실패를 포함한 전체 평균 block 방문 | 24.07 | 22.95 |
| 첫 target 직접 성공 | 42.48% | 27.73% |

이 평가는 원문 정답을 보고 분기를 선택하는 oracle이며, iteration 예산 후 실패한 target의 정답을 입력으로 공급하여 다음 target 평가를 계속한다. 30.97%는 정답을 모르는 자유 생성의 정확도가 아니다. 전체 block 방문 4.65% 감소는 prefill·LM Head·주입·cache 비용을 포함한 FLOPs 또는 실제 추론 시간의 절감률을 뜻하지 않는다.

현재 학습 코드는 원본 weight를 고정하고 q/v LoRA, token entry/memory/carry gate, wrap, exit 보정 파라미터를 업데이트한다. Transformer 본체 전체 미세조정과 배포형 exit gate는 연구계획의 후속 항목이다. 7B 확대 실험은 데이터 준비 기록만 있고 완료된 학습 결과는 없다.

- [0.5B 확대 결과](results/training_parallel_05b/comparison.md)
- [학습 완료량·학습 범위·GPU 동기화 기록](results/training_parallel_05b/training/done.json)
- [전체 실행 해석](PARALLEL_TRAINING.md)
- [모든 보존 실험의 상태](results/index.json)

## 실험 계열과 실행기

| 계열 | 실행기 | 확인하는 것 |
| --- | --- | --- |
| 표준 forward의 layer별 readout | `run.py` | 중간 표현의 최초 정답 깊이와 readout 학습 효과 |
| 실제 중간 주입·순환, 고정 방문 예산 | `run_routed.py` | 성공 시 같은 layer에 다음 token 주입, 실패 시 다음 layer 및 wrap |
| 누적 iteration 학습 | `run_training_revision.py` | index+1 제한, 실패 label의 loss 유지와 다음 token 학습 계속 |
| 동기식 다중 GPU 학습 | `run_training_parallel.py` | 서로 다른 episode 계산 후 방문 수로 가중한 gradient 동기화 |
| 제한 없는 loop 관측 | `run_loop_pilot.py`, `run_loop_pilot_tokens.py` | 강제 정답 입력 없이 완주·검열·loop 수 관측 |

최초 무학습 관측은 24층 모델에서 8,192 targets 중 52.77%에 정답이 나타났고, 이 집합의 평균 최초 정답 깊이는 21.30층이었다. 초기 readout 학습 결과는 `results/pilot_05b/`, `results/main_7b/`에 과거 실험으로 보존한다. 이 분석과 실제 순환 경로의 학습·가속 효과를 구분한다.

## 환경과 모델 준비

Python 3.10 이상과 CUDA GPU 환경을 사용한다. 실제 검증 버전은 [requirements-tested.txt](requirements-tested.txt)에 기록했다. GPU 선택 정책은 원래 실험 서버의 물리 번호 **1·2·3만 허용**한다. 각 프로세스의 `cuda:0`은 UUID로 제한한 뒤의 논리 번호다.

```bash
git clone https://github.com/KimHyoungchan/Intermediate-Generate-LM.git
cd Intermediate-Generate-LM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-tested.txt
pip install -e .
```

설정은 모델의 로컬 cache를 사용하므로 최초 실행 전에 공개 모델을 내려받는다. 7B는 해당 config의 모델 ID와 revision을 사용한다.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
)
PY
```

WikiText는 prepare 단계에서 고정 revision을 로드한다. 다른 서버에서 기존 checkpoint를 가지고 있어도 config/source/data identity가 맞아야 사용할 수 있다. 이 저장소의 요약 결과만으로 학습 후 추론을 실행할 수는 없으며, 동일 설정으로 데이터를 준비하고 학습해야 한다.

## 실행

가장 작은 다중 GPU 동작 확인:

```bash
python run_training_parallel.py --config configs/training_parallel_smoke.yaml --gpus 1,2,3
```

0.5B 확대 학습 전후 비교:

```bash
python run_training_parallel.py --config configs/training_parallel_05b.yaml --gpus 1,2,3
```

각 명령은 데이터 준비 → 학습 전 평가 → 학습 → 학습 후 평가 → 비교를 수행한다. 학습은 여러 GPU를 사용하고 validation 및 학습 전후 평가는 선택한 첫 GPU에서 수행한다. 완료 단계 재사용은 `--resume`, 다른 실험 결과 경로는 `--output-dir`을 사용한다. 중단된 학습의 optimizer step 복원은 지원하지 않는다.

단일 GPU 누적 iteration 실험과 초기 readout 분석:

```bash
python run_training_revision.py --config configs/training_revision_05b.yaml --gpu 1
python run.py --config configs/pilot_05b.yaml --gpu 1
```

96회 block 방문 예산의 순환 실험과 이후 loop 관측:

```bash
python run_routed.py --config configs/routed_05b_v96.yaml --gpu 1
python run_loop_pilot_tokens.py --config configs/loop_pilot_05b.yaml --gpu 1
```

Loop 관측은 source experiment의 데이터와 checkpoint가 필요하다. 시간 내 미완주는 관측한 loop 수의 하한으로 기록한다. `--max-seconds 0`은 시간 watchdog도 해제하므로 완주하거나 직접 중단할 때까지 실행한다.

## 검사

다음 검사는 GPU와 사전학습 모델 다운로드 없이 작은 CPU 모델과 Gloo 통신으로 실행한다.

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s tests_routed -v
python -m unittest discover -s tests_training_revision -v
python -m unittest discover -s tests_loop_pilot -v
python -m unittest discover -s tests_parallel_training -v
```

검사 범위는 label 정렬, 같은-layer 재진입, wrap 상태와 KV 교체, 이후 token loss의 앞 상태 전달, 불균등한 rank별 방문 수의 gradient 가중 평균, 마지막 sample의 중복·누락 방지를 포함한다.

## 연구 문서와 결과 보존

- [원안과 연구 목표](RESEARCH_GOALS.md)
- [실제 순환 구조](ROUTED_EXPERIMENT.md), [누적 iteration 학습](TRAINING_REVISION.md)
- [데이터 전체 규모와 실제 사용량](DATASET_SCALE.md)
- [Loop 측정과 검열 해석](LOOP_PILOT.md)
- [제출용 연구계획서: Word](proposal/LLM_Inference_Acceleration_Research_Proposal.docx), [PDF](proposal/LLM_Inference_Acceleration_Research_Proposal.pdf)
- [결과 탐색 안내](results/README.md)

모델·데이터·소스 hash와 원래 실행 설정은 결과 snapshot에 보존했다. 로컬의 원본 실험 폴더는 변경하지 않는다. 상세 trace·학습 데이터·checkpoint가 필요한 재분석은 원래 실행 자료 또는 새 재현 실행을 사용한다. Model 및 dataset의 이용 조건은 각 배포처를 따른다.
