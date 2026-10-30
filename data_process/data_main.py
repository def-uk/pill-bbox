"""데이터 파이프라인 진입점. 원본 JSON에서 YOLO 학습 입력(data.yaml)까지 0–4단계를 한 번에 만든다.

    uv run python -m data_process.data_main --label v1 --split combo_v1_s42 setup30fix_v1_s42

산출물은 모두 data/processed/ 아래에 쓴다(docs/DATA_PIPELINE_PLAN.md 5장).
"""

import argparse
import json
import random
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 그림을 파일로만 저장한다. 화면이 없는 서버·Colab에서도 돈다.
import matplotlib.pyplot as plt
import pandas as pd

from data_process.data_processing import (
    list_images,
    load_annotations,
    measure_bbox_padding,
    read_image_headers,
)
from data_process.data_yaml import (
    build_class_map,
    class_names,
    data_yaml_path,
    write_fold_files,
    yolo_dir,
)
from data_process.download_data import download_data
from data_process.make_labels import (
    LABEL_VERSIONS,
    PADDING_MIN,
    PADDING_REVIEW_NAME,
    build_corrections,
    check_labels,
    file_sha256,
    files_sha256,
    load_padding_review,
    make_label_version,
    merge_padding_review,
    padding_review_table,
    read_yolo_labels,
    save_corrections,
    write_yolo_labels,
)
from data_process.move_files import link_images
from data_process.split import build_split, parse_split_name, save_split
from src.config import PROJECT_ROOT
from visualization import (
    draw_bbox,
    draw_bboxes,
    image_grid,
    save_figure,
    set_korean_font,
    show_image,
    zoom_to,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
# EDA 9.1의 규모. 다르면 경고만 하고 진행한다.
EXPECTED_COUNTS = {"train_images": 232, "train_annotations": 763, "test_images": 842}
EXPECTED_LABELS = {"raw": 763, "v1": 771, "v2": 771, "v2auto": 771}
PADDING_DIR = PROCESSED_DIR / "padding"  # 여백 측정 전체 결과와 검수용 그림 (커밋 안 함)
REVIEW_PAGE_SIZE = 20  # 검수 그림 한 장에 넣을 박스 수
# 0이 아니면 학습에 쓸 수 없는 라벨 검사 항목 (raw 버전은 기록만 한다)
BLOCKING_CHECKS = ["bbox_missing", "out_of_image", "non_positive", "duplicate_label"]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def find_raw_dir(raw_dir: Path = RAW_DIR) -> Path:
    """압축을 푼 위치에 따라 sprint_ai_project1_data 폴더가 한 단계 더 있으면 그 폴더를 쓴다."""
    nested = raw_dir / "sprint_ai_project1_data"
    return nested if nested.is_dir() else raw_dir


def step0_data() -> Path:
    """[0] 원본이 없으면 받고, 이미지·JSON 수를 세어 EDA와 다르면 경고한다. 원본 폴더를 반환한다."""
    if not (find_raw_dir() / "train_images").is_dir():
        download_data()
    raw_dir = find_raw_dir()
    counts = {
        "train_images": len(list_images(raw_dir / "train_images")),
        "train_annotations": len(list((raw_dir / "train_annotations").rglob("*.json"))),
        "test_images": len(list_images(raw_dir / "test_images")),
    }
    for key, expected in EXPECTED_COUNTS.items():
        if counts[key] != expected:
            warnings.warn(f"{key} {counts[key]}개 (EDA {expected}개)", stacklevel=2)
    print(f"[0] 원본: {counts}")
    return raw_dir


def step1_load(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """[1] 라벨을 읽고 실제 이미지 크기를 헤더로 확인한다. (ann_df, train 이미지 크기, 요약)."""
    ann_df, error_df = load_annotations(raw_dir / "train_annotations")
    if error_df["excluded"].any():
        warnings.warn(f"읽지 못해 빠진 라벨이 있습니다:\n{error_df}", stacklevel=2)
    train_sizes = read_image_headers(list_images(raw_dir / "train_images"), "train")
    test_sizes = read_image_headers(list_images(raw_dir / "test_images"), "test")
    json_sizes = ann_df.groupby("file_name")[["width", "height"]].first()
    header = train_sizes.set_index("file_name")[["width", "height"]].reindex(json_sizes.index)
    size_mismatch = sorted(json_sizes.index[(json_sizes != header).any(axis=1)])
    if size_mismatch:
        warnings.warn(f"JSON 크기와 실제 크기가 다른 사진 {size_mismatch}", stacklevel=2)
    summary = {
        "n_annotations": len(ann_df),
        "load_errors": len(error_df),
        "size_mismatch": size_mismatch,
        "header_errors": int(pd.concat([train_sizes, test_sizes])["error"].notna().sum()),
        "train_resolutions": _resolutions(train_sizes),
        "test_resolutions": _resolutions(test_sizes),
    }
    print(f"[1] 라벨 {len(ann_df)}개, 크기 불일치 {len(size_mismatch)}장")
    print(f"    해상도 train {summary['train_resolutions']} / test {summary['test_resolutions']}")
    return ann_df, train_sizes, summary


def _resolutions(sizes: pd.DataFrame) -> dict[str, int]:
    keys = sizes["width"].astype(str) + "x" + sizes["height"].astype(str)
    return {key: int(n) for key, n in keys.value_counts().items()}


def step2_labels(
    ann_df: pd.DataFrame, train_sizes: pd.DataFrame, version: str, raw_dir: Path
) -> tuple[pd.DataFrame, str]:
    """[2] 보정 기록과 라벨 버전을 저장한다. (라벨 표, 라벨 파일 SHA256)을 반환한다."""
    ann_dir = raw_dir / "train_annotations"
    corrections = build_corrections(ann_df, ann_dir, PROCESSED_DIR)
    save_corrections(corrections, PROCESSED_DIR / "corrections.csv")
    labels, log = make_label_version(ann_df, corrections, version, train_sizes)
    checks = check_labels(labels)
    padding = corrections[corrections["stage"] == "여백 보정"]["reviewed"]
    padding_counts = {
        "approved": int(padding.eq(True).sum()),
        "rejected": int(padding.eq(False).sum()),
        "pending": int(padding.isna().sum()),
    }
    if version == "v2":
        print(f"[2] 여백 검수: {padding_counts} (승인만 v2에 적용)")
        if not padding_counts["approved"]:
            warnings.warn("승인된 여백 보정이 없어 v2가 v1과 같습니다", stacklevel=2)
    if version == "v2auto":
        print(f"[2] 여백 검수: {padding_counts} (거부만 빼고 미검수까지 v2auto에 적용)")
    if version != "raw":
        problems = {key: checks[key] for key in BLOCKING_CHECKS if checks[key]}
        if problems:
            raise ValueError(f"{version} 라벨 검사 실패: {problems}")
    if version in EXPECTED_LABELS and checks["n_labels"] != EXPECTED_LABELS[version]:
        warnings.warn(f"{version} 라벨 {checks['n_labels']}개", stacklevel=2)

    labels_path = PROCESSED_DIR / "labels" / f"labels_{version}.csv"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(labels_path, index=False, encoding="utf-8")
    labels_sha256 = file_sha256(labels_path)
    meta = {
        "label_version": version,
        "labels_sha256": labels_sha256,
        "input_json_sha256": files_sha256(list(ann_dir.rglob("*.json")), ann_dir),
        "corrections_sha256": file_sha256(PROCESSED_DIR / "corrections.csv"),
        "applied_corrections": log["correction_id"].tolist(),
        "padding_review": padding_counts,
        "checks": checks,
        "created_at": _now(),
    }
    _write_json(labels_path.with_suffix(".json"), meta)
    print(f"[2] {labels_path.relative_to(PROJECT_ROOT)}: {checks}, 보정 {len(log)}건 적용")
    return labels, labels_sha256


def step3_split(labels: pd.DataFrame, labels_sha256: str, split_name: str) -> pd.DataFrame:
    """[3] 분할을 만들고 fold_k.csv와 meta.json(누수 검사 포함)을 저장한다."""
    spec = parse_split_name(split_name)
    folds, meta = build_split(labels, spec, labels_sha256)
    out_dir = PROCESSED_DIR / "splits" / split_name
    save_split(folds, out_dir)
    _write_json(out_dir / "meta.json", meta)
    checks = pd.DataFrame(meta["checks"]).set_index("fold")
    checks["val_only_classes"] = checks["val_only_classes"].map(len)
    print(f"[3] {split_name}: 학습 고정 사진 {meta['fixed_train_images']}장")
    print(checks.round(2).to_string())
    return folds


def step4_yolo(
    labels: pd.DataFrame,
    version: str,
    raw_dir: Path,
    splits: dict[str, pd.DataFrame],
    use_code_names: bool,
    n_check: int,
    seed: int,
) -> list[Path]:
    """[4] 이미지 링크, YOLO 라벨, class_map, fold별 txt·data.yaml을 만들고 변환을 검사한다."""
    root = yolo_dir(version)
    train_paths = list_images(raw_dir / "train_images")
    links = link_images(train_paths, root / "images" / "train")
    test_links = link_images(list_images(raw_dir / "test_images"), root / "images" / "test")
    print(f"[4] 이미지 train {links}, test {test_links}")

    class_map = build_class_map(labels)
    class_map.to_csv(root / "class_map.csv", index=False, encoding="utf-8")
    class_idx = dict(zip(class_map["category_id"], class_map["yolo_idx"], strict=True))
    file_names = [p.name for p in train_paths]
    label_dir = root / "labels" / "train"
    result = write_yolo_labels(labels, file_names, class_idx, label_dir)
    checks = {**result, **check_yolo_labels(labels, label_dir, class_idx)}
    _write_json(root / "check.json", {"label_version": version, **checks, "created_at": _now()})
    print(f"    YOLO 라벨 검사: {checks}")
    if not (checks["coords_in_range"] and checks["classes_in_range"]):
        raise ValueError(f"YOLO 라벨 범위 오류: {checks}")
    if checks["n_lines"] != len(labels) - checks["n_dropped"]:
        raise ValueError(f"YOLO 라벨 수가 다릅니다: {checks['n_lines']} / {len(labels)}")
    if checks["n_empty"]:
        warnings.warn(f"bbox 없는 사진 {checks['n_empty']}장 (빈 txt)", stacklevel=2)
    save_visual_check(labels, root, class_idx, n_check, seed)

    names = class_names(class_map, use_code=use_code_names)
    yaml_paths = []
    for split_name, folds in splits.items():
        for fold, rows in folds.groupby("fold"):
            header = {"label_version": version, "split_name": split_name, "fold": fold}
            out_dir = data_yaml_path(version, split_name, int(fold)).parent
            yaml_paths.append(
                write_fold_files(rows, root / "images" / "train", out_dir, names, header)
            )
    print(f"    data.yaml {len(yaml_paths)}개: {yaml_paths[0].relative_to(PROJECT_ROOT)} 등")
    return yaml_paths


def check_yolo_labels(
    labels: pd.DataFrame, label_dir: Path, class_idx: dict[int, int]
) -> dict[str, object]:
    """txt를 다시 읽어 좌표·클래스 범위와 픽셀 역변환 오차(이미지 안으로 자른 원본 기준)를 잰다."""
    errors, coords_ok, classes_ok = [], True, True
    for file_name, boxes in labels.groupby("file_name"):
        width, height = boxes["width"].iloc[0], boxes["height"].iloc[0]
        txt = label_dir / Path(file_name).with_suffix(".txt").name
        values = [float(v) for line in txt.read_text().splitlines() for v in line.split()[1:]]
        coords_ok &= all(0 <= v <= 1 for v in values)
        back = read_yolo_labels(txt, width, height)
        classes_ok &= bool(back["yolo_idx"].between(0, len(class_idx) - 1).all())
        x1, y1 = boxes["x"].clip(0, width), boxes["y"].clip(0, height)
        x2 = (boxes["x"] + boxes["w"]).clip(0, width)
        y2 = (boxes["y"] + boxes["h"]).clip(0, height)
        clipped = pd.DataFrame(
            {
                "yolo_idx": boxes["category_id"].map(class_idx),
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
            }
        )
        clipped = clipped[(clipped["w"] > 0) & (clipped["h"] > 0)]
        if len(clipped) != len(back):
            errors.append(float("inf"))
            continue
        a = clipped.sort_values(["yolo_idx", "x", "y"]).to_numpy(dtype=float)
        b = back.sort_values(["yolo_idx", "x", "y"]).to_numpy(dtype=float)
        errors.append(float(abs(a - b).max()) if len(a) else 0.0)
    return {
        "coords_in_range": coords_ok,
        "classes_in_range": classes_ok,
        "roundtrip_max_px": max(errors, default=0.0),
    }


def save_visual_check(
    labels: pd.DataFrame, root: Path, class_idx: dict[int, int], n_check: int, seed: int
) -> Path | None:
    """무작위 사진에 원본 박스(빨강)와 txt에서 되돌린 박스(초록 점선)를 겹쳐 check/에 저장한다."""
    if n_check <= 0:
        return None
    files = sorted(labels["file_name"].unique())
    picks = random.Random(seed).sample(files, min(n_check, len(files)))
    idx_to_cat = {idx: cat for cat, idx in class_idx.items()}
    fig, axes = image_grid(len(picks), cols=min(len(picks), 5))
    for ax, file_name in zip(axes, picks, strict=True):
        boxes = labels[labels["file_name"] == file_name]
        width, height = boxes["width"].iloc[0], boxes["height"].iloc[0]
        txt = root / "labels" / "train" / Path(file_name).with_suffix(".txt").name
        back = read_yolo_labels(txt, width, height)
        back["category_id"] = back["yolo_idx"].map(idx_to_cat)
        show_image(ax, root / "images" / "train" / file_name, file_name.split("_")[0], fontsize=7)
        draw_bboxes(ax, boxes, color="red")
        draw_bboxes(ax, back, label_col=None, color="lime", linestyle="--")
    out_path = root / "check" / "yolo_roundtrip.png"
    save_figure(fig, out_path)
    plt.close(fig)
    print(f"    시각 확인: {out_path.relative_to(PROJECT_ROOT)} ({len(picks)}장)")
    return out_path


def prepare_padding_review(padding_min: float = PADDING_MIN) -> Path:
    """v1 라벨로 여백을 재고 검수표와 비교 그림을 만든다. 검수표 경로를 반환한다.

    이미 있는 검수표의 decision·reviewer·comment는 old/new bbox가 같은 행에 그대로 옮긴다.
    """
    raw_dir = step0_data()
    ann_df, train_sizes, _ = step1_load(raw_dir)
    labels, _ = step2_labels(ann_df, train_sizes, "v1", raw_dir)
    print("[여백] GrabCut 측정 중 (수 분 걸린다)")
    padding_df = measure_bbox_padding(labels, raw_dir / "train_images")
    PADDING_DIR.mkdir(parents=True, exist_ok=True)
    padding_df.to_csv(PADDING_DIR / "measure_v1.csv", index=False, encoding="utf-8")

    review = padding_review_table(padding_df, labels, padding_min)
    review_path = PROCESSED_DIR / PADDING_REVIEW_NAME
    if review_path.is_file():
        review = merge_padding_review(review, load_padding_review(review_path))
    review.to_csv(review_path, index=False, encoding="utf-8")
    over = padding_df["padding_ratio"] >= padding_min
    manual = padding_df[over & (padding_df["status"] != "estimated")]
    manual.to_csv(PADDING_DIR / "needs_manual_v1.csv", index=False, encoding="utf-8")
    print(f"[여백] 상태별 박스 수: {padding_df['status'].value_counts().to_dict()}")
    print(
        f"[여백] 검수표 {review_path.relative_to(PROJECT_ROOT)}: 후보 {len(review)}개 "
        f"(여백 {padding_min:.0%} 이상), 이미 결정된 행 {int((review['decision'] != '').sum())}개"
    )
    print(f"[여백] 자동 추정이 불안정해 후보에서 뺀 박스 (여백 기준 이상): {len(manual)}개")
    save_review_pages(review, labels, raw_dir / "train_images")
    return review_path


def save_review_pages(review: pd.DataFrame, labels: pd.DataFrame, img_dir: Path) -> list[Path]:
    """검수표 순서대로 원본 박스(빨강)와 제안 박스(초록 점선)를 확대해 그림 여러 장으로 저장한다."""
    set_korean_font()
    out_dir = PADDING_DIR / "review_pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("page_*.png"):
        old.unlink()
    sizes = labels.groupby("file_name")[["width", "height"]].first()
    paths = []
    for start in range(0, len(review), REVIEW_PAGE_SIZE):
        page = review.iloc[start : start + REVIEW_PAGE_SIZE]
        fig, axes = image_grid(len(page), cols=5, cell=(3.2, 3.8))
        for ax, row in zip(axes, page.itertuples(), strict=True):
            old_bbox, new_bbox = json.loads(row.old_bbox), json.loads(row.new_bbox)
            title = f"{row.review_id} {row.category_id} 여백 {row.padding_ratio:.0%}"
            show_image(ax, img_dir / row.file_name, title, fontsize=8)
            draw_bbox(ax, old_bbox, color="red")
            draw_bbox(ax, new_bbox, color="lime", linestyle="--")
            zoom_to(ax, [old_bbox, new_bbox], margin=0.25, limit=tuple(sizes.loc[row.file_name]))
        path = out_dir / f"page_{start // REVIEW_PAGE_SIZE + 1:02d}.png"
        save_figure(fig, path)
        plt.close(fig)
        paths.append(path)
    print(f"[여백] 검수 그림 {len(paths)}장: {out_dir.relative_to(PROJECT_ROOT)}")
    return paths


def run(
    label_version: str,
    split_names: list[str],
    use_code_names: bool = False,
    n_check: int = 5,
    seed: int = 42,
) -> list[Path]:
    """0–4단계를 차례로 실행하고 만든 data.yaml 경로 목록을 반환한다."""
    if label_version not in LABEL_VERSIONS:
        raise ValueError(f"라벨 버전 {label_version}이 없습니다: {list(LABEL_VERSIONS)}")
    for split_name in split_names:
        split_label = parse_split_name(split_name).label_version
        if split_label != label_version:
            raise ValueError(f"{split_name}은 {split_label} 라벨용입니다 (--label {label_version})")
    raw_dir = step0_data()
    ann_df, train_sizes, summary = step1_load(raw_dir)
    _write_json(PROCESSED_DIR / "raw_check.json", {**summary, "created_at": _now()})
    labels, labels_sha256 = step2_labels(ann_df, train_sizes, label_version, raw_dir)
    splits = {name: step3_split(labels, labels_sha256, name) for name in split_names}
    return step4_yolo(labels, label_version, raw_dir, splits, use_code_names, n_check, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="원본 → 라벨 버전 → 분할 → YOLO 데이터")
    parser.add_argument("--label", default="v1", choices=list(LABEL_VERSIONS))
    parser.add_argument(
        "--split",
        nargs="+",
        help="분할 이름. 기본은 combo_<label>_s42, setup30fix_<label>_s42 (계획 3단계)",
    )
    parser.add_argument("--code-names", action="store_true", help="names를 category_id로 쓴다")
    parser.add_argument("--n-check", type=int, default=5, help="시각 확인할 사진 수 (0이면 생략)")
    parser.add_argument("--seed", type=int, default=42, help="시각 확인 사진을 고르는 seed")
    parser.add_argument(
        "--prepare-padding-review",
        action="store_true",
        help="v1 라벨로 여백을 재고 검수표·검수 그림만 만든다 (분할·YOLO 변환 안 함)",
    )
    parser.add_argument("--padding-min", type=float, default=PADDING_MIN, help="여백 후보 기준")
    args = parser.parse_args()
    if args.prepare_padding_review:
        prepare_padding_review(args.padding_min)
        return
    split_names = args.split or [f"combo_{args.label}_s42", f"setup30fix_{args.label}_s42"]
    run(args.label, split_names, args.code_names, args.n_check, args.seed)


if __name__ == "__main__":
    main()
