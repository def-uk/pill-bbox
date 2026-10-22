# AGENTS.md

AI 코딩 에이전트(Codex, Claude Code 등)가 이 저장소에서 작업할 때 따르는 지침입니다.
에이전트 지침은 이 파일 한 곳에서 관리합니다. `CLAUDE.md`는 이 파일을 불러오기만 하므로, 규칙을 바꿀 때는 이 파일을 고칩니다.

## 프로젝트

- 알약 이미지에서 알약 위치(bbox)와 종류를 검출하는 프로젝트입니다.
- 검출 모델은 아직 정하지 않았습니다. 여러 모델을 실험해 본 뒤 확정합니다. 특정 모델을 전제로 구조를 바꾸지 않습니다.
- 초기 단계입니다. `data_process/`, `models/`의 `.py` 파일과 `main.py`는 아직 비어 있는 뼈대입니다. 파일 이름과 폴더 구조는 팀이 정한 것이므로 임의로 옮기거나 합치지 않습니다.
- 사람용 문서: [README.md](README.md), 코드 규칙 [docs/CODE_CONVENTION.md](docs/CODE_CONVENTION.md), Git 규칙 [docs/GIT_CONVENTION.md](docs/GIT_CONVENTION.md), 데이터 구조 [data/README.md](data/README.md). 아래 요약과 다르면 docs가 우선입니다.

## 폴더

| 경로 | 역할 |
|---|---|
| `src/config.py` | 공통 설정: `PROJECT_ROOT`, `get_device()`, `init_wandb()` |
| `data_process/` | 전처리, 라벨 생성(`make_labels.py`), 파일 이동, `data.yaml` 생성 |
| `models/` | 학습(`train.py`), 평가(`evaluate.py`), 튜닝(`tuning.py`, optuna) |
| `data/` | 원본 데이터. `train_images/`, `train_annotations/`(JSON), `test_images/`는 커밋되지 않음 |
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

## Git 규칙 (요약)

- `main`에서 커밋하거나 `main`으로 push하지 않습니다. 브랜치 이름은 `<type>/<짧은-설명>`.
- 커밋 메시지는 `<type>: <요약>`. type은 `feat`, `fix`, `exp`, `data`, `refactor`, `docs`, `chore` 중 하나. 요약은 한국어 50자 이내, 마침표 없음. 커밋 하나에 변경 하나.
- PR 제목도 같은 형식. 본문은 `.github/PULL_REQUEST_TEMPLATE.md`를 채웁니다(`gh pr create --fill` 금지).

## 에이전트 작업 원칙

- 사용자가 요청하지 않으면 커밋, push, PR 생성을 하지 않습니다.
- 커밋하면 안 되는 것: `data/` 아래 이미지·annotation, `*.pt`, `runs/`, `wandb/`, `.env`, API 키. 이런 파일을 만들거나 수정했다면 스테이징하지 않습니다.
- 전체 학습(긴 epoch, 튜닝)은 요청이 있을 때만 실행합니다. 동작 확인은 `epochs=1`, 작은 `fraction` 등 짧게 돌리고 `WANDB_MODE=disabled`를 붙입니다.
- `data/` 아래 이미지와 JSON은 수가 많고 큽니다. 전체를 읽지 말고 파일 몇 개나 `data/README.md`의 스키마로 구조를 파악합니다.
- 노트북(`.ipynb`)을 고쳤다면 출력을 지운 상태로 둡니다. 두 번 이상 쓰는 코드는 `.py`로 옮겨 import합니다.
- 패키지가 필요하면 `uv add`로 추가하고, 왜 필요한지 사용자에게 알립니다.
- 문서와 실제 파일이 다르면 조용히 한쪽에 맞추지 말고 사용자에게 알립니다.
- 컨벤션을 바꾸는 작업이면 `docs/`와 이 파일의 요약을 함께 고칩니다.
