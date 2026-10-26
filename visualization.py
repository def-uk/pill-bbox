from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

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
