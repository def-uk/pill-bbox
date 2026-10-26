import math
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from PIL import Image

# 먼저 찾는 순서: Windows → Mac → Linux(나눔) → Linux(Noto)
KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]
# WSL에서는 Windows 폰트 파일을 직접 등록해야 matplotlib이 찾는다.
WSL_MALGUN = Path("/mnt/c/Windows/Fonts/malgun.ttf")


def set_korean_font() -> str | None:
    """설치된 한글 폰트를 matplotlib 기본 폰트로 정하고 그 이름을 반환한다. 없으면 None."""
    if WSL_MALGUN.exists():
        fm.fontManager.addfont(str(WSL_MALGUN))
    installed = {f.name for f in fm.fontManager.ttflist}
    # 한글 폰트에서 마이너스 기호가 깨지지 않게 한다.
    plt.rcParams["axes.unicode_minus"] = False
    for font in KOREAN_FONTS:
        if font in installed:
            plt.rcParams["font.family"] = font
            return font
    return None


def image_grid(n: int, cols: int = 4, cell: tuple[float, float] = (4, 5.5)) -> tuple:
    """이미지 n장을 놓을 격자를 만든다. 남는 칸은 숨기고 (fig, 사용할 axes 목록)을 반환한다."""
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols) if n else 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell[0], rows * cell[1]), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    return fig, list(axes.flat[:n])


def show_image(ax: Axes, img_path: Path, title: str | None = None, fontsize: int = 8) -> None:
    """이미지를 축에 그린다. 축 범위를 이미지 크기로 고정해 밖에 있는 bbox는 잘려 보인다."""
    with Image.open(img_path) as img:
        ax.imshow(img.convert("RGB"))
    ax.set_autoscale_on(False)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize)


def draw_bboxes(
    ax: Axes,
    boxes: pd.DataFrame,
    label_col: str | None = "category_id",
    color: str = "red",
    linestyle: str = "-",
) -> None:
    """x, y, w, h 열이 있는 표의 bbox를 축에 그린다. 좌표가 NaN인 행은 건너뛴다."""
    for row in boxes.dropna(subset=["x", "y", "w", "h"]).itertuples():
        ax.add_patch(
            Rectangle((row.x, row.y), row.w, row.h, fill=False, edgecolor=color, lw=2, ls=linestyle)
        )
        if label_col:
            ax.text(
                row.x,
                row.y - 8,
                str(getattr(row, label_col)),
                color="white",
                fontsize=7,
                va="bottom",
                bbox={"facecolor": color, "alpha": 0.7, "edgecolor": "none", "pad": 1},
            )
