import math
from collections.abc import Iterable, Sequence
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


def show_image(
    ax: Axes,
    img_path: Path,
    title: str | None = None,
    fontsize: int = 8,
    rotate180: bool = False,
) -> None:
    """이미지를 축에 그린다. 축 범위를 이미지 크기로 고정해 밖에 있는 bbox는 잘려 보인다.

    rotate180=True면 이미지를 180° 돌려 그린다. bbox는 rotate_bboxes_180으로 함께 돌린다.
    """
    with Image.open(img_path) as img:
        rgb = img.convert("RGB")
        ax.imshow(rgb.rotate(180) if rotate180 else rgb)
    ax.set_autoscale_on(False)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize)


def draw_bbox(
    ax: Axes,
    bbox: Sequence[float],
    label: str | None = None,
    color: str = "red",
    linestyle: str = "-",
) -> None:
    """[x, y, w, h] 박스 하나를 축에 그리고 왼쪽 위에 label을 붙인다."""
    x, y, w, h = bbox
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, lw=2, ls=linestyle))
    if label:
        ax.text(
            x,
            y - 8,
            label,
            color="white",
            fontsize=7,
            va="bottom",
            clip_on=True,
            bbox={"facecolor": color, "alpha": 0.7, "edgecolor": "none", "pad": 1},
        )


def draw_bboxes(
    ax: Axes,
    boxes: pd.DataFrame,
    label_col: str | None = "category_id",
    color: str = "red",
    linestyle: str = "-",
) -> None:
    """x, y, w, h 열이 있는 표의 bbox를 축에 그린다. 좌표가 NaN인 행은 건너뛴다."""
    for row in boxes.dropna(subset=["x", "y", "w", "h"]).itertuples():
        label = str(getattr(row, label_col)) if label_col else None
        draw_bbox(ax, (row.x, row.y, row.w, row.h), label, color, linestyle)


def draw_image_bounds(ax: Axes, width: float, height: float) -> None:
    """이미지 경계를 초록 점선으로 그린다. 이미지 밖 좌표를 볼 때 기준선으로 쓴다."""
    ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor="lime", lw=1.5, ls="--"))


def zoom_to(
    ax: Axes,
    bboxes: Iterable[Sequence[float]],
    margin: float = 0.15,
    limit: tuple[float, float] | None = None,
) -> None:
    """[x, y, w, h] 박스들을 모두 포함하도록 축 범위를 바꾼다. 여백은 가장 긴 변 × margin이다.

    limit=(width, height)를 주면 이미지 안으로 자르고, None이면 이미지 밖 좌표까지 보여 준다.
    """
    boxes = [tuple(map(float, b)) for b in bboxes]
    boxes = [b for b in boxes if not any(math.isnan(v) for v in b)]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    pad = max(x1 - x0, y1 - y0) * margin
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    if limit:
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, limit[0]), min(y1, limit[1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)


def plot_corr_heatmap(ax: Axes, corr: pd.DataFrame, title: str):
    """상관계수 행렬을 −1~1 색으로 칠하고 칸마다 값을 적는다. 색 막대용 mesh를 반환한다."""
    mesh = ax.pcolormesh(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, edgecolors="white", lw=2)
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            value = corr.iat[i, j]
            color = "white" if abs(value) >= 0.6 else "#1f1f1f"
            ax.text(j + 0.5, i + 0.5, f"{value:.2f}", ha="center", va="center", color=color)
    ax.set_xticks(
        [k + 0.5 for k in range(len(corr.columns))], corr.columns, rotation=45, ha="right"
    )
    ax.set_yticks([k + 0.5 for k in range(len(corr))], corr.index)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    ax.set_frame_on(False)
    ax.set_title(title, fontsize=11)
    return mesh
