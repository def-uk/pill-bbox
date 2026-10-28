# AGENTS.md

AI 코딩 에이전트(Codex, Claude Code 등)가 이 저장소에서 작업할 때 따르는 지침입니다.
에이전트 지침은 이 파일 한 곳에서 관리합니다. `CLAUDE.md`는 이 파일을 불러오기만 하므로, 규칙을 바꿀 때는 이 파일을 고칩니다.

## 프로젝트

- 알약 이미지에서 알약 위치(bbox)와 종류를 검출하는 프로젝트입니다.
- 검출 모델은 아직 확정하지 않았습니다. 후보 YOLO11m·YOLO26m을 같은 데이터로 비교하는 중입니다. 데이터 파이프라인(라벨·분할·YOLO 변환)은 특정 모델을 전제로 바꾸지 않습니다.
- 데이터 파이프라인(`data_process/`)과 학습·평가(`models/train.py`, `models/evaluate.py`)는 구현되어 있습니다. `main.py`, `models/tuning.py`는 아직 비어 있는 뼈대입니다. 파일 이름과 폴더 구조는 팀이 정한 것이므로 임의로 옮기거나 합치지 않습니다.
- 사람용 문서: [README.md](README.md), 코드 규칙 [docs/CODE_CONVENTION.md](docs/CODE_CONVENTION.md), Git 규칙 [docs/GIT_CONVENTION.md](docs/GIT_CONVENTION.md), 데이터 구조 [data/README.md](data/README.md), 데이터 파이프라인 사용법 [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md). 아래 요약과 다르면 docs가 우선입니다.

## 폴더

| 경로 | 역할 |
|---|---|
| `src/config.py` | 공통 설정: `PROJECT_ROOT`, `RUNS_DIR`, `get_device()`, `init_wandb()` |
| `data_process/` | 데이터 파이프라인. 진입점 `data_main.py`, 보정·라벨 버전(`make_labels.py`), 분할(`split.py`), 이미지 링크(`move_files.py`), `data.yaml` 생성(`data_yaml.py`) |
| `models/` | 학습(`train.py`, 진입점 `model_main.py`), 평가·test 예측(`evaluate.py`), 튜닝(`tuning.py`, optuna) |
| `data/` | `raw/`는 원본 이미지·JSON, `processed/`는 파이프라인 산출물·보정 기록. `processed/` 바로 아래 csv만 커밋 가능 |
| `runs/` | 학습 결과(ultralytics)와 평가 결과(`runs/<exp>/results/`). 커밋되지 않음. Colab에서는 `--runs-dir`로 Drive에 둠 |
| `scripts/` | 개발 보조 스크립트. `md_lock.py`는 md 문서 잠금 |
| `notebooks/` | 탐색·시각화 노트북 |
| `team_test/` | 팀원 개인 Git 연습용. 건드리지 않음 |

## 명령

패키지와 실행은 모두 `uv`로 합니다. `pip install`, `python -m venv`는 쓰지 않습니다.

```bash
uv sync                                   # 환경 맞추기
uv run python main.py                     # 실행
uv add <패키지> / uv add --dev <패키지>     # 패키지 추가 (pyproject.toml, uv.lock 함께 커밋)
uv run ruff format . && uv run ruff check . --fix   # 코드를 고친 뒤 항상 실행
WANDB_MODE=disabled uv run python <파일>   # 에이전트가 확인용으로 실행할 때

# 데이터 파이프라인 (자세히: docs/DATA_PIPELINE.md)
uv run python -m data_process.data_main --label v1          # 원본 → 라벨 → 분할 → data.yaml
uv run python -m data_process.data_main --prepare-padding-review   # 여백 검수표·그림 (약 3분)
WANDB_MODE=disabled uv run python -m models.model_main --model yolo11m.pt yolo26m.pt \
    --split combo_v1_s42 --folds 0 --epochs 1 --fraction 0.1 --exp smoke   # 학습 동작 확인
```

테스트 스위트는 아직 없습니다. 바꾼 코드는 ruff 통과와 작은 입력으로 직접 실행해 확인합니다.

## 코드 규칙 (요약)

- Python 3.12. ruff 규칙은 `pyproject.toml`에만 있습니다. 한 줄 100자, 큰따옴표.
- 이름: 파일·함수·변수 `snake_case`, 클래스 `PascalCase`, 상수 `UPPER_SNAKE_CASE`. 축약은 `img`, `bbox`, `cfg` 정도만.
- import는 프로젝트 루트 기준 절대 경로(`from src.config import get_device`). `import *` 금지.
- 함수 인자와 반환값에 타입 힌트. 다른 파일에서 쓰는 함수에는 docstring(한국어 한 줄 이상).
- 경로는 `pathlib.Path`와 `src/config.py`의 `PROJECT_ROOT` 기준. 개인 PC 절대 경로(`/Users/...`, `C:\...`) 금지.
- 장치는 `"cuda"`, `"mps"`를 직접 쓰지 않고 `get_device()`. Mac(mps)과 NVIDIA(cuda)에서 같은 코드가 돌아야 합니다.
- 하이퍼파라미터는 함수 인자나 설정으로 받고, 실험마다 seed를 고정해 기록합니다.
- wandb 기록은 `init_wandb(model, ...)`로만 켭니다. ultralytics 전역 설정(`settings.update(wandb=True)` 등)은 바꾸지 않습니다.
- 주석과 docstring은 한국어로 씁니다.

## 데이터 파이프라인

- 학습 데이터는 `data_process.data_main`으로만 만듭니다. 라벨 txt, `data.yaml`, 분할 목록을 손으로 만들거나 노트북에서 따로 만들지 않습니다. 노트북은 `data_process.data_yaml.data_yaml_path(라벨 버전, 분할, fold)`로 경로를 받아 씁니다.
- 기본 라벨은 `v1`, 기본 분할은 `combo_v1_s42`와 `setup30fix_v1_s42`입니다. 분할 이름 형식은 `<단위>_<라벨 버전>_s<seed>`이고 라벨 버전과 맞아야 합니다.
- 라벨을 고쳐야 하면 원본 JSON이나 `labels_*.csv`를 직접 고치지 않습니다. 보정 기록(`make_labels.py`의 기록 생성 함수, `data/processed/*.csv`)을 바꾸고 다시 실행합니다. 새 보정은 `reviewed`(사람 검수)가 있어야 적용됩니다.
- 여백 검수표 `data/processed/padding_review.csv`의 `decision`(승인/거부)은 사람이 채웁니다. 에이전트가 대신 승인하지 않습니다.
- `data.yaml`과 txt 목록에는 그 PC의 절대 경로가 들어갑니다. 다른 PC·Colab으로 복사하지 말고 그곳에서 다시 실행합니다.
- 모델 비교는 한 fold 점수가 아니라 같은 분할의 5 fold 평균·표준편차로 합니다. 주 지표는 대회 채점 지표와 같은 mAP75-95(`models/evaluate.py`의 `MAIN_METRIC`)입니다. 학습은 전체 파인튜닝(`freeze` 없음)이 기본이고, 증강은 `--aug on off`, 옵티마이저는 `--optimizer`(기본 `auto` = AdamW)로 비교합니다.
- Mac(MPS)에서는 `epochs=1`, 작은 `fraction`으로 동작 확인만 합니다. 본 학습은 사용자가 요청할 때 GPU(Colab 등)에서 합니다.
- 파이프라인 코드를 고치면 `--label v1`을 다시 실행해 자동 검사(누수 0%, 라벨 검사, YOLO 변환 검사)가 통과하는지 확인합니다.

## 하이퍼파라미터

에이전트는 하이퍼파라미터를 혼자 정하지 않습니다. 팀원마다 다른 값이 말없이 섞이면 실험끼리 비교할 수 없기 때문입니다.

- 학습을 실행하거나, 학습·튜닝 코드를 짜거나 고치기 전에 아래 기본값을 표로 보여 주고 바꿀 항목이 있는지 사용자에게 묻습니다. 기본값과 다른 값을 제안할 때는 이유를 한 줄 붙입니다.
- 사용자가 답하기 전에는 학습을 실행하지 않습니다. 동작 확인(`epochs=1`, 작은 `fraction`, `WANDB_MODE=disabled`)은 기본값으로 바로 해도 됩니다.
- 정한 값은 코드에 고정하지 않고 `model_main` 인자나 `train()` 인자로 넘깁니다. 실행한 명령 전체를 사용자에게 알려 같은 명령으로 다시 돌릴 수 있게 합니다.
- 기본값 자체(아래 표, `AUG_PRESETS`, `MAIN_METRIC`, 함수 기본 인자)를 바꾸려면 사용자 확인을 받고 이 표와 `docs/DATA_PIPELINE.md`를 함께 고칩니다.

| 항목 | 기본값 | 인자 | 비고 |
|---|---|---|---|
| 모델 | `yolo11m.pt`, `yolo26m.pt` | `--model` | 후보 비교 중. 전체 파인튜닝(`freeze` 없음) |
| 라벨 | `v1` | `--label` | `v2`는 여백 검수 승인분이 있을 때만 의미 있음 |
| 분할 | `setup30fix_v1_s42`, `combo_v1_s42` | `--split` | 어느 분할로 고를지 사용자에게 확인. 비교는 5 fold 전부 |
| epochs | 100 | `--epochs` | 필수 인자 |
| 이미지 크기 / 배치 | 640 / 16 | `--imgsz`, `--batch` | GPU 메모리가 부족하면 배치를 줄임 |
| seed | 42 | `--seed` | 실험마다 기록 |
| 증강 | `on` | `--aug` | 값은 `models/train.py`의 `AUG_PRESETS`. 비교는 `--aug on off` |
| 옵티마이저 | `auto` (= AdamW, lr0 약 0.00017) | `--optimizer`, `--lr0`, `--cos-lr` | 직접 고를 때 lr0 기본값은 SGD 계열 0.01, 그 밖 0.001 |
| 주 지표 | mAP75-95 | `MAIN_METRIC` | best.pt 선택·early stopping 기준. 대회 채점 지표 mAP@[0.75:0.95]와 같음 |
| 그 밖 | ultralytics 기본값 (`patience` 100 등) | `train(..., patience=30)` | `model_main`에 없는 값은 Python에서 `train()` 인자로 넘김 |

## Git 규칙 (요약)

- `main`에서 커밋하거나 `main`으로 push하지 않습니다. 브랜치 이름은 `<type>/<짧은-설명>`.
- 커밋 메시지는 `<type>: <요약>`. type은 `feat`, `fix`, `exp`, `data`, `refactor`, `docs`, `chore` 중 하나. 요약은 한국어 50자 이내, 마침표 없음. 커밋 하나에 변경 하나.
- PR 제목도 같은 형식. 본문은 `.github/PULL_REQUEST_TEMPLATE.md`를 채웁니다(`gh pr create --fill` 금지).

## md 문서 잠금

저장소의 `.md` 파일(이 파일 포함)은 기본적으로 잠겨 있습니다. 에이전트가 사용자 허락 없이 문서를 바꾸지 않게 하기 위해서입니다.

- 에이전트는 잠금을 직접 풀거나 우회하지 않습니다. `md_lock.py unlock` 실행, `chmod`, `.md_unlocked` 파일 생성, 다른 경로로 덮어쓰기 모두 금지입니다.
- md 파일을 고쳐야 하면 작업을 멈추고, 고칠 파일과 이유를 알린 뒤 사용자에게 터미널에서 아래 명령을 실행해 달라고 요청합니다.

  ```bash
  uv run python scripts/md_lock.py unlock
  ```

- 사용자가 해제했다고 알려 주면 수정합니다. md 수정을 모두 마치면 마지막 응답에서 다시 잠그는 명령을 안내합니다.

  ```bash
  uv run python scripts/md_lock.py lock
  ```

- 상태 확인: `uv run python scripts/md_lock.py status`. 잠긴 동안 md 파일은 읽기 전용이고, Claude Code는 hook이 수정을 거부합니다.

## 에이전트 작업 원칙

- 사용자가 요청하지 않으면 커밋, push, PR 생성을 하지 않습니다.
- 커밋하면 안 되는 것: `data/` 아래 이미지·annotation·파이프라인 산출물(`data/processed/` 하위 폴더), `*.pt`, `runs/`, `wandb/`, `.env`, API 키. 이런 파일을 만들거나 수정했다면 스테이징하지 않습니다.
- 전체 학습(긴 epoch, 튜닝)은 요청이 있을 때만 실행합니다. 동작 확인은 `epochs=1`, 작은 `fraction` 등 짧게 돌리고 `WANDB_MODE=disabled`를 붙입니다.
- `data/` 아래 이미지와 JSON은 수가 많고 큽니다. 전체를 읽지 말고 파일 몇 개나 `data/README.md`의 스키마로 구조를 파악합니다.
- 노트북(`.ipynb`)을 고쳤다면 출력을 지운 상태로 둡니다. 두 번 이상 쓰는 코드는 `.py`로 옮겨 import합니다.
- 패키지가 필요하면 `uv add`로 추가하고, 왜 필요한지 사용자에게 알립니다.
- 문서와 실제 파일이 다르면 조용히 한쪽에 맞추지 말고 사용자에게 알립니다.
- 컨벤션을 바꾸는 작업이면 `docs/`와 이 파일의 요약을 함께 고칩니다.
