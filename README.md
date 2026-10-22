# pillbbox

알약 이미지에서 알약의 위치(bbox)와 종류를 검출하는 프로젝트입니다.

- 입력: 알약 촬영 이미지 / 출력: 알약별 bbox와 클래스
- 모델: 미정. 여러 모델을 실험해 본 뒤 확정합니다.
- 실험 기록: [wandb](#실험-기록-wandb)

<!-- 채워 주세요: 목표 지표(예: test mAP50-95), 데이터 출처, 일정, 팀원과 역할 -->

## 문서

| 문서 | 내용 |
|---|---|
| [docs/CODE_CONVENTION.md](docs/CODE_CONVENTION.md) | 코드 컨벤션 (포맷, 이름, import, 경로·장치) |
| [docs/GIT_CONVENTION.md](docs/GIT_CONVENTION.md) | Git 컨벤션 (커밋 메시지, 브랜치, PR) |
| [data/README.md](data/README.md) | annotation JSON 구조 |
| [AGENTS.md](AGENTS.md) | AI 코딩 에이전트(Codex, Claude Code) 지침 |

## 치트시트

자주 쓰는 명령만 모았습니다. 자세한 설명은 아래 각 섹션에 있습니다.

| 상황 | 명령 |
|---|---|
| 처음 받았을 때 (한 번) | `uv sync` → `git config core.hooksPath .githooks` |
| pull 받은 뒤 | `uv sync` |
| 코드 실행 | `uv run python main.py` |
| 노트북 열기 | `uv run jupyter lab` |
| 패키지 추가 / 제거 | `uv add <패키지>` / `uv remove <패키지>` |
| 개발용 패키지 추가 | `uv add --dev <패키지>` |
| 커밋 전 코드 정리 | `uv run ruff format .` → `uv run ruff check . --fix` |
| Mac에서 MPS 에러 | `PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python main.py` |

작업 흐름:

```bash
git switch main && git pull && uv sync      # 최신 상태로 맞추기
git switch -c feat/<짧은-설명>                # 브랜치 만들기
# ... 작업 ...
uv run ruff format . && uv run ruff check . --fix
git add <파일> && git commit -m "feat: <요약>"
git push -u origin feat/<짧은-설명>
gh pr create --base main                     # PR 만들기 (아래 "PR 올리기" 참고)
```

- 패키지를 바꿨다면 `pyproject.toml`과 `uv.lock`을 함께 커밋합니다.
- 노트북은 출력을 지운 뒤 커밋합니다.
- `pip install`은 쓰지 않습니다.

## 개발 환경

이 프로젝트는 [uv](https://docs.astral.sh/uv/)로 Python 버전과 패키지를 관리합니다.

- Python 3.12 (`.python-version`에 고정)
- 패키지 목록: `pyproject.toml`
- 정확한 설치 버전: `uv.lock`

### 1. uv 설치

```bash
# macOS
brew install uv

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 환경 만들기

```bash
uv sync
```

`.venv`가 만들어지고 `uv.lock`에 적힌 버전 그대로 설치됩니다. Python 3.12가 없으면 uv가 알아서 받아 옵니다.

같이 git hook도 켭니다. `main`에 직접 커밋·push하는 실수를 막아 줍니다.

```bash
git config core.hooksPath .githooks
```

OS마다 PyTorch가 다르게 설치됩니다.

| OS | PyTorch | 사용 장치 |
|---|---|---|
| macOS | PyPI 기본 빌드 | `mps` (Apple GPU) |
| Windows / Linux | CUDA 12.6 빌드 | `cuda` (NVIDIA GPU) |

NVIDIA GPU를 쓰려면 CUDA 12.6을 지원하는 드라이버가 설치되어 있어야 합니다.

### 3. 실행

```bash
uv run python main.py
uv run jupyter lab
```

`uv run`을 붙이면 가상환경을 따로 활성화하지 않아도 됩니다. VS Code에서는 인터프리터 / 커널로 `.venv`를 선택하세요.

## 패키지 관리

```bash
uv add <패키지>          # 추가
uv add "<패키지>>=1.2"   # 버전 범위 지정
uv add --dev <패키지>    # 개발용 (jupyter, linter 등)
uv remove <패키지>       # 제거
```

- 패키지를 추가하거나 제거했다면 **`pyproject.toml`과 `uv.lock`을 함께 커밋**하세요.
- pull 받은 뒤 다른 사람이 패키지를 바꿨다면 `uv sync`를 다시 실행하세요.
- `pip install`은 쓰지 마세요. 설치는 되지만 `pyproject.toml`에 기록되지 않아 다른 팀원 환경에 반영되지 않습니다.

## 장치 선택

`src/config.py`의 `get_device()`가 실행 환경에 맞는 장치를 골라 줍니다 (cuda → mps → cpu 순).

```python
from src.config import get_device

model.train(..., device=get_device())
```

macOS에서 MPS가 지원하지 않는 연산 때문에 에러가 나면, 그 연산만 CPU로 돌리도록 실행하세요.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python main.py
```

## 실험 기록 (wandb)

처음 한 번, `.env.example`을 `.env`로 복사하고 [API 키](https://wandb.ai/authorize)를 채웁니다. `.env`는 커밋되지 않습니다.

```bash
cp .env.example .env   # WANDB_API_KEY, WANDB_ENTITY(팀 이름) 입력
```

학습 전에 모델을 넘겨 `init_wandb()`를 호출하면 그 모델의 학습 로그(loss, mAP, 그래프, 가중치)가 wandb에 기록됩니다. ultralytics 전역 설정은 건드리지 않으므로 이 프로젝트 밖의 YOLO 학습에는 영향이 없습니다.

```python
from ultralytics import YOLO

from src.config import get_device, init_wandb

model = YOLO("yolov8s.pt")
init_wandb(model, run_name="yolov8s-baseline", config={"seed": 42})
model.train(data="data.yaml", epochs=50, device=get_device())
```

| 상황 | 명령 |
|---|---|
| wandb 끄고 실행 | `WANDB_MODE=disabled uv run python main.py` |
| 인터넷 없이 기록만 남기기 | `WANDB_MODE=offline uv run python main.py` |
| offline 기록 나중에 올리기 | `uv run wandb sync wandb/offline-run-*` |

## 코드 컨벤션

[docs/CODE_CONVENTION.md](docs/CODE_CONVENTION.md)를 참고하세요. 커밋 전에 아래 두 명령을 실행합니다.

```bash
uv run ruff format .
uv run ruff check . --fix
```

VS Code에서는 추천 확장(Ruff)을 설치하면 저장할 때 자동으로 정리됩니다. ruff 규칙은 `pyproject.toml`에만 둡니다.

## Git 컨벤션

자세한 내용은 [docs/GIT_CONVENTION.md](docs/GIT_CONVENTION.md)를 참고하세요. 핵심만 요약하면:

- `main`에 직접 커밋·push하지 않고, `<type>/<짧은-설명>` 브랜치를 만들어 PR로 합칩니다.
- 커밋 메시지는 `<type>: <요약>` (한국어, 50자 이내, 마침표 없음). type은 `feat` `fix` `exp` `data` `refactor` `docs` `chore`.
- PR 본문은 [PR 템플릿](.github/PULL_REQUEST_TEMPLATE.md)을 채웁니다.

## AI 코딩 에이전트

Codex, Claude Code 등 에이전트용 지침은 [AGENTS.md](AGENTS.md) 하나로 관리합니다.

- Codex는 `AGENTS.md`를 바로 읽습니다.
- Claude Code는 `CLAUDE.md`를 읽고, `CLAUDE.md`가 `AGENTS.md`를 불러옵니다.
- 규칙을 바꿀 때는 `AGENTS.md`만 고칩니다. 개인 설정은 `CLAUDE.local.md` 등 커밋되지 않는 파일에 둡니다.
- 에이전트가 만든 코드도 커밋 전에 직접 읽고, 위 컨벤션과 PR 템플릿 체크리스트를 똑같이 적용합니다.

## 폴더 구조

```
pillbbox/
├── data/               # 데이터 (구조: data/README.md, 이미지·annotation은 커밋 안 함)
├── docs/               # 컨벤션 등 문서
├── data_process/       # 전처리, 라벨 생성, 데이터셋 yaml 생성
├── models/             # 학습, 평가, 튜닝
├── notebooks/          # 실험 노트북
├── team_test/          # 팀원 Git 연습용
├── src/                # 공통 설정 (config.py)
├── AGENTS.md           # AI 에이전트 지침 (CLAUDE.md는 이 파일을 불러옴)
├── CLAUDE.md
├── main.py
├── visualization.py
├── pyproject.toml
└── uv.lock
```
