# pillbbox

알약 검출 프로젝트

## 치트시트

자주 쓰는 명령만 모았습니다. 자세한 설명은 아래 각 섹션에 있습니다.

| 상황 | 명령 |
|---|---|
| 처음 받았을 때 / pull 받은 뒤 | `uv sync` |
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

## Git 커밋 컨벤션

### 형식

```
<type>: <요약>

본문(선택)
```

- 요약은 한국어로, 50자 이내로, 끝에 마침표 없이 씁니다.
- 관련 이슈가 있으면 끝에 `#이슈번호`를 붙입니다.
- 커밋 하나에는 한 가지 변경만 담습니다.

### type

| type | 언제 | 예시 |
|---|---|---|
| `feat` | 새 기능 | `feat: YOLO 라벨 변환 스크립트 추가` |
| `fix` | 버그 수정 | `fix: bbox 좌표가 이미지 밖으로 나가는 문제 수정` |
| `exp` | 실험 설정, 하이퍼파라미터 변경 | `exp: yolov8s, imgsz 640으로 학습` |
| `data` | 데이터 전처리, 분할 방식 변경 | `data: train/val 분할 비율 8:2로 변경` |
| `refactor` | 동작은 그대로, 구조만 정리 | `refactor: 경로 설정을 config.py로 이동` |
| `docs` | 문서 | `docs: README에 실행 방법 추가` |
| `chore` | 패키지, 설정, 기타 | `chore: optuna 의존성 추가` |

### 예시

```
exp: yolov8m 기본 설정으로 50 epoch 학습

yolov8s 대비 작은 알약 검출이 부족해서 모델 크기를 키움.
val mAP50: 0.81 -> 0.86

#12
```

### 브랜치

`main`에 바로 push하지 않고, 브랜치를 만들어 PR로 합칩니다.

```
<type>/<짧은-설명>      예: feat/label-converter, exp/yolov8m-baseline
```

### PR 올리기

브랜치를 push한 뒤 GitHub CLI(`gh`)로 PR을 만듭니다.

```bash
# 처음 한 번: 설치와 로그인
brew install gh                    # macOS
winget install --id GitHub.cli     # Windows
sudo apt install gh                # Linux/Ubuntu
gh auth login

# PR 만들기
git push -u origin feat/<짧은-설명>
gh pr create --base main           # 제목·본문을 묻고, 본문에는 PR 템플릿이 채워짐
gh pr create --base main --web     # 브라우저에서 작성하고 싶을 때

# PR 확인
gh pr status                       # 내 PR 상태
gh pr view --web                   # 현재 브랜치의 PR을 브라우저로 열기
```

- PR 제목은 커밋 형식과 같이 `<type>: <요약>`으로 씁니다.
- 본문은 [PR 템플릿](.github/PULL_REQUEST_TEMPLATE.md)을 채웁니다. `--fill`을 쓰면 템플릿이 빠지므로 쓰지 않습니다.
- `gh`를 쓰지 않는다면 push 후 GitHub 저장소 페이지에 뜨는 **Compare & pull request** 버튼을 눌러도 됩니다.

### 최신 버전 받기

PR이 머지된 뒤 새 작업을 시작할 때는 `main`을 최신으로 받고 새 브랜치를 만듭니다. 머지된 브랜치에서 이어서 작업하지 않습니다.

```bash
git switch main
git pull origin main
git switch -c feat/<짧은-설명>
```

작업 중인 브랜치에 `main`의 최신 변경을 반영할 때는 브랜치를 옮기지 않고 그대로 받습니다.

```bash
git pull origin main               # main의 변경을 현재 브랜치에 merge

# 커밋하지 않은 작업이 있어 pull이 막힐 때
git stash                          # 작업 중인 변경을 임시 저장
git pull origin main
git stash pop                      # 작업 내용 되돌리기
```

- 충돌이 나면 해당 파일의 `<<<<<<<` ~ `>>>>>>>` 구간을 정리하고 `git add <파일>` 후 커밋합니다.
- `git stash pop`에서 충돌이 나면 stash가 남아 있으므로, 정리한 뒤 `git stash drop`으로 지웁니다.

## 폴더 구조

```
pillbbox/
├── data/               # 데이터 (설명: data/data.txt)
├── docs/               # 컨벤션 등 문서
├── data_process/       # 전처리, 라벨 생성, YOLO yaml 생성
├── models/             # 학습, 평가, 튜닝
├── notebooks/          # 실험 노트북
├── src/                # 공통 설정 (config.py)
├── main.py
├── visualization.py
├── pyproject.toml
└── uv.lock
```
