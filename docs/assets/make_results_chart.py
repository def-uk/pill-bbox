"""README 결과 막대그래프(SVG, 라이트·다크)를 만든다.

점수를 바꾸거나 최종 모델 점수를 채우면 RESULTS를 고치고 다시 실행한다.

    uv run python docs/assets/make_results_chart.py
"""

from pathlib import Path

OUT_DIR = Path(__file__).parent

# (모델, 설정, Kaggle mAP@[0.75:0.95], 종류)
# 종류: "base" 베이스라인(회색), "changed" 바꾼 설정(강조), "final" 최종 모델(점수 None이면 점선)
# 바꾼 설정은 점수 옆 배지에 같은 모델 베이스라인 대비 증가율을 적는다.
RESULTS = [
    ("YOLO11m", "베이스라인", 0.29946, "base"),
    ("YOLO11m", "외부 데이터 + 외부 클래스까지 학습", 0.63060, "changed"),
    ("YOLO26m", "베이스라인", 0.17877, "base"),
    ("YOLO26m", "외부 데이터 + copy-paste", 0.607, "changed"),
    ("Faster R-CNN", "베이스라인", 0.25918, "base"),
    ("Faster R-CNN", "외부 데이터 + copy-paste", 0.45127, "changed"),
    ("YOLO11m", "최종 모델 (전체 데이터 학습)", None, "final"),
]

THEMES = {
    "light": {
        "text": "#1f2328",
        "muted": "#59636e",
        "grid": "#d1d9e0",
        "base": "#8c8b85",
        "changed": "#452AF7",
        "pill_bg": "#ECE9FE",
        "pill_text": "#3A22D6",
    },
    "dark": {
        "text": "#f0f6fc",
        "muted": "#9198a1",
        "grid": "#3d444d",
        "base": "#6e6d68",
        "changed": "#8270FF",
        "pill_bg": "#2B2558",
        "pill_text": "#B9AFFF",
    },
}

FONT = (
    "-apple-system, BlinkMacSystemFont, 'Apple SD Gothic Neo', 'Malgun Gothic', "
    "'Noto Sans KR', 'Segoe UI', sans-serif"
)
LEGEND = (("base", "베이스라인 (30 에폭, 외부 데이터 없음)"), ("changed", "바꾼 설정"))

WIDTH = 820
LABEL_W = 300  # 왼쪽 라벨 영역
PLOT_W = 390
ROW_H = 34
BAR_H = 20
GROUP_GAP = 14
TOP = 56
X_MAX = 0.7
RADIUS = 4
PILL_H = 18


def text_width(text, size):
    """SVG 글자 폭 어림값. 한글은 글자 크기만큼, 나머지는 그 절반 남짓."""
    return sum(size * (0.95 if ord(ch) > 0x1100 else 0.6) for ch in text)


def pill(x, cy, label, c):
    """증가율 배지. x는 왼쪽 끝, cy는 글자 기준선."""
    w = text_width(label, 11.5) + 14
    y = cy - 4.5 - PILL_H / 2 + 0.5
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{PILL_H}" '
        f'rx="{PILL_H / 2}" fill="{c["pill_bg"]}"/>'
        f'<text x="{x + w / 2:.1f}" y="{cy - 0.5:.1f}" font-size="11.5" font-weight="700" '
        f'fill="{c["pill_text"]}" text-anchor="middle">{label}</text>'
    ), w


def bar_path(x0, y, w, h, r):
    """왼쪽(기준선)은 각지고 오른쪽 끝만 둥근 막대."""
    r = min(r, w, h / 2)
    return (
        f"M{x0:.1f},{y:.1f} H{x0 + w - r:.1f} "
        f"Q{x0 + w:.1f},{y:.1f} {x0 + w:.1f},{y + r:.1f} "
        f"V{y + h - r:.1f} Q{x0 + w:.1f},{y + h:.1f} {x0 + w - r:.1f},{y + h:.1f} "
        f"H{x0:.1f} Z"
    )


def render(theme):
    c = THEMES[theme]
    x0 = LABEL_W
    sx = PLOT_W / X_MAX

    # 행 위치: 모델이 바뀌면 간격을 둔다
    rows, y, prev = [], TOP, None
    for model, setting, score, kind in RESULTS:
        if prev is not None and model != prev:
            y += GROUP_GAP
        rows.append((y, model, setting, score, kind, model != prev))
        y += ROW_H
        prev = model
    plot_bottom = y
    height = plot_bottom + 34
    baselines = {m: s for m, _, s, k in RESULTS if k == "base"}

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" font-family="{FONT}" role="img" '
        f'aria-label="모델·설정별 Kaggle mAP@[0.75:0.95] 막대그래프">'
    ]

    # 범례: 색 두 가지와 증가율 배지
    lx = 16
    for key, label in LEGEND:
        out.append(f'<rect x="{lx}" y="14" width="12" height="12" rx="3" fill="{c[key]}"/>')
        out.append(
            f'<text x="{lx + 18}" y="24.5" font-size="13" fill="{c["muted"]}">{label}</text>'
        )
        lx += 18 + text_width(label, 13) + 22
    svg, pw = pill(lx, 24.5, "+%", c)
    out.append(svg)
    out.append(
        f'<text x="{lx + pw + 6:.1f}" y="24.5" font-size="13" fill="{c["muted"]}">'
        "같은 모델 베이스라인 대비</text>"
    )

    # 격자와 축 눈금
    for i in range(int(X_MAX * 10) + 1):
        v = i / 10
        gx = x0 + v * sx
        dash = "" if i == 0 else ' stroke-dasharray="2 4"'
        out.append(
            f'<line x1="{gx:.1f}" y1="{TOP - 8}" x2="{gx:.1f}" y2="{plot_bottom}" '
            f'stroke="{c["grid"]}" stroke-width="1"{dash}/>'
        )
        out.append(
            f'<text x="{gx:.1f}" y="{plot_bottom + 18}" font-size="12" fill="{c["muted"]}" '
            f'text-anchor="middle">{v:.1f}</text>'
        )

    for ry, model, setting, score, kind, first in rows:
        by = ry + (ROW_H - BAR_H) / 2
        cy = ry + ROW_H / 2 + 4.5
        if first:
            out.append(
                f'<text x="16" y="{cy}" font-size="14" font-weight="600" '
                f'fill="{c["text"]}">{model}</text>'
            )
        out.append(
            f'<text x="{x0 - 12}" y="{cy}" font-size="13" fill="{c["muted"]}" '
            f'text-anchor="end">{setting}</text>'
        )
        if score is None:
            w = PLOT_W * 0.5
            out.append(
                f'<rect x="{x0 + 0.5}" y="{by + 0.5}" width="{w}" height="{BAR_H - 1}" '
                f'rx="{RADIUS}" '
                f'fill="none" stroke="{c["changed"]}" stroke-width="1.5" stroke-dasharray="4 3"/>'
            )
            out.append(
                f'<text x="{x0 + 10}" y="{cy}" font-size="12" '
                f'fill="{c["muted"]}">결과 기입 예정</text>'
            )
            continue
        w = score * sx
        fill = c["base"] if kind == "base" else c["changed"]
        weight = "600" if kind != "base" else "400"
        out.append(f'<path d="{bar_path(x0, by, w, BAR_H, RADIUS)}" fill="{fill}"/>')
        label = f"{score:.3f}"
        out.append(
            f'<text x="{x0 + w + 8:.1f}" y="{cy}" font-size="13" font-weight="{weight}" '
            f'fill="{c["text"]}">{label}</text>'
        )
        base = baselines.get(model)
        if kind != "base" and base:
            px = x0 + w + 8 + text_width(label, 13) + 8
            out.append(pill(px, cy, f"{score / base - 1:+.0%}", c)[0])
    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    for theme in THEMES:
        path = OUT_DIR / f"results_{theme}.svg"
        path.write_text(render(theme), encoding="utf-8")
        print(path)
