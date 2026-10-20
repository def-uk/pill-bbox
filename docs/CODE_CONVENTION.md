# 코드 컨벤션

## 포맷팅과 린트

[ruff](https://docs.astral.sh/ruff/)로 통일합니다. 규칙은 `pyproject.toml`의 `[tool.ruff]` 한 곳에만 둡니다. 에디터 설정이나 개인 설정에 ruff 규칙을 따로 넣지 마세요.

```bash
uv run ruff format .        # 코드 정리
uv run ruff check . --fix   # 린트 + import 정렬 자동 수정
```

- 커밋 전에 두 명령을 실행하세요.
- 한 줄은 최대 100자입니다.
- 따옴표는 큰따옴표(`"`)를 씁니다. `ruff format`이 맞춰 줍니다.
- VS Code에서 프로젝트를 열면 Ruff 확장 설치를 권하는 알림이 뜹니다. 설치하면 저장할 때 자동으로 포맷과 import 정렬이 됩니다(`.vscode/settings.json`).

## 이름 짓기

| 대상 | 규칙 | 예시 |
|---|---|---|
| 파일, 모듈 | `snake_case` | `make_labels.py` |
| 함수, 변수 | `snake_case` | `load_annotations()`, `image_dir` |
| 클래스 | `PascalCase` | `PillDataset` |
| 상수 | `UPPER_SNAKE_CASE` | `IMAGE_SIZE = 640` |

- 축약어는 널리 쓰이는 것(`img`, `bbox`, `cfg`)만 씁니다.
- 반복문 변수가 아니라면 한 글자 이름(`a`, `x`)은 쓰지 않습니다.

## import

- 프로젝트 루트 기준의 절대 경로로 가져옵니다.
  ```python
  from src.config import get_device
  from data_process.make_labels import make_labels
  ```
- 순서는 표준 라이브러리 → 외부 패키지 → 프로젝트 코드입니다. `ruff check --fix`가 정렬해 줍니다.
- `from x import *`는 쓰지 않습니다.

## 함수

- 함수의 인자와 반환값에는 타입 힌트를 붙입니다.
- 다른 파일에서 가져다 쓰는 함수에는 한 줄 이상의 docstring을 씁니다.

```python
def make_labels(csv_path: Path, output_dir: Path) -> int:
    """annotation CSV를 YOLO 라벨(txt)로 변환하고, 만든 파일 수를 반환한다."""
```

- 함수 하나는 한 가지 일만 하게 만듭니다. 50줄을 넘기면 나눌 수 있는지 살펴보세요.

## 경로와 설정

- 경로는 `pathlib.Path`로 다룹니다.
- `/Users/...`, `C:\...` 같은 개인 PC의 절대 경로는 코드에 쓰지 않습니다. 기준 경로는 `src/config.py`에 모아 두고 가져다 씁니다.
- 하이퍼파라미터(epoch, batch size, learning rate 등)는 코드 곳곳에 흩어 두지 말고 설정 파일이나 함수 인자로 받습니다.
- 장치는 `"cuda"`처럼 직접 쓰지 말고 `get_device()`를 씁니다. Mac(mps)과 NVIDIA PC(cuda)에서 같은 코드가 돌아가야 합니다.
- 결과를 재현할 수 있도록 실험마다 seed를 고정하고 기록합니다.

## 노트북

- 노트북은 탐색과 시각화용입니다. 두 번 이상 쓰는 코드는 `.py`로 옮기고 import해서 씁니다.
- 커밋 전에 출력을 지웁니다(Clear All Outputs). 출력이 남아 있으면 파일이 커지고 diff를 읽을 수 없습니다.
- 파일 이름은 `번호_내용.ipynb` 형식으로 짓습니다(예: `01_eda.ipynb`).

## 커밋하지 않는 것

`.gitignore`에 등록되어 있습니다.

- 데이터: `data/` 아래 파일(`data/data.txt` 제외)
- 모델 가중치: `*.pt`
- 학습 결과: `runs/`, `wandb/`
- 가상환경: `.venv/`
- API 키, 토큰: wandb 키 등은 환경 변수로 넘기고 코드에 쓰지 않습니다.
