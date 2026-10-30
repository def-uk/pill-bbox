import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from data_process.data_processing import BBOX_COLS, PADDING_VERSION

# 보정 기록 한 행의 열. 기존 라벨은 (source_json, annotation_id)로, 새 라벨은 새 annotation_id로
# 가리킨다. input_sha256은 기록을 만들 때 본 원본 JSON의 해시, map_status는 대상 라벨 확정 여부다.
CORRECTION_COLS = [
    "correction_id",
    "stage",
    "case",
    "source_json",
    "file_name",
    "category_id",
    "annotation_id",
    "old_bbox",
    "new_bbox",
    "method",
    "author",
    "reviewed",
    "source_record",
    "input_sha256",
    "map_status",
]
# 적용 순서. 앞 단계가 바꾼 좌표가 뒤 단계 기록의 old_bbox와 다르면 뒤 단계 기록은 거부된다.
STAGES = ["추가", "위치 수정", "여백 보정"]
# 새 라벨 annotation_id의 시작값. 원본 ID와 겹치지 않도록 크게 둔다.
NEW_ID_START = 1_000_000
# 적용 결과 표의 열. status: 적용, 이미 적용, 미검수, 거부
APPLY_COLS = ["correction_id", "stage", "case", "file_name", "category_id", "status", "reasons"]
# eda_kwon.ipynb fixed_annotations.csv의 case 이름 → 통합 기록의 case 이름
KWON_CASES = {
    "좌표가 이미지 밖": "이미지 밖 좌표",
    "빈 배경에 박스": "빈 배경 박스",
    "다른 약과 좌표 중복": "다른 약과 좌표 중복",
}
# 새 라벨에 원본 값을 복사하지 않는 열. 나머지는 같은 사진·같은 약의 값을 복사한다.
_LABEL_COLS = ["source_json", "annotation_id", "category_id", *BBOX_COLS, "area"]
_ADDED_COLS = ["label_origin", "correction_ids"]


def _as_bbox(values: Sequence[float]) -> tuple[int, int, int, int]:
    """bbox 값을 비교하기 쉬운 정수 튜플로 바꾼다."""
    x, y, w, h = (int(round(float(v))) for v in values)
    return x, y, w, h


def file_sha256(path: Path) -> str:
    """파일 바이트의 SHA256을 반환한다. 보정 기록이 어느 원본을 보고 만든 것인지 남긴다."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def correction_for_label(
    ann_df: pd.DataFrame,
    source_json: str,
    annotation_id: int,
    new_bbox: Sequence[float],
    ann_dir: Path,
    **fields: object,
) -> dict:
    """라벨 하나의 bbox를 고치는 보정 기록 한 행을 만든다. old_bbox는 ann_df의 현재 값이다.

    fields에는 correction_id, stage, case, method, author, reviewed, source_record를 넘긴다.
    source_json이 None이면 '추가' 단계로 새로 생긴 라벨(새 annotation_id)을 가리킨다.
    """
    target = ann_df[_label_mask(ann_df, source_json, annotation_id)]
    if len(target) != 1:
        raise ValueError(f"대상 라벨이 {len(target)}개입니다: {source_json} #{annotation_id}")
    row = target.iloc[0]
    record = dict.fromkeys(CORRECTION_COLS)
    record.update(fields)
    record.update(
        source_json=source_json,
        file_name=row["file_name"],
        category_id=int(row["category_id"]),
        annotation_id=int(annotation_id),
        old_bbox=_as_bbox(row[BBOX_COLS]),
        new_bbox=_as_bbox(new_bbox),
        input_sha256=None if pd.isna(source_json) else file_sha256(ann_dir / source_json),
        map_status="확정",
    )
    return record


def _label_mask(df: pd.DataFrame, source_json: object, annotation_id: object) -> pd.Series:
    """(source_json, annotation_id)로 라벨을 고른다. source_json이 비면 새로 추가된 라벨이다."""
    same_source = (
        df["source_json"].isna() if pd.isna(source_json) else df["source_json"] == source_json
    )
    return same_source & (df["annotation_id"] == annotation_id)


def corrections_from_kwon(
    added_path: Path, fixed_path: Path, ann_df: pd.DataFrame, ann_dir: Path, reviewed: bool
) -> pd.DataFrame:
    """eda_kwon.ipynb의 added·fixed CSV(셀 19, 78)를 통합 보정 기록으로 바꾼다.

    kwon CSV는 라벨을 (file_name, category_id)로 가리키므로 원본 라벨로 역매핑한다. 후보가
    하나가 아니면 source_json·annotation_id를 비우고 map_status에 후보 수를 적는다(적용 시 거부).
    old_bbox는 CSV의 old_* 값(kwon이 고칠 때 본 좌표)이다. 새 라벨의 annotation_id는 비워 두고
    assign_new_ids로 붙인다. 없는 파일은 건너뛴다.
    """
    records = []
    if added_path.is_file():
        for i, row in pd.read_csv(added_path).iterrows():
            record = dict.fromkeys(CORRECTION_COLS)
            stem = Path(row["file_name"]).stem
            record.update(
                correction_id=f"kwon_added:{stem}:{row['category_id']}",
                stage="추가",
                case="라벨 누락",
                file_name=row["file_name"],
                category_id=int(row["category_id"]),
                new_bbox=_as_bbox(row[BBOX_COLS]),
                method="OpenCV 배경 차이로 박스 없는 알약 검출 + 사방 8px",
                author="gwon753-dev",
                reviewed=reviewed,
                source_record=f"{added_path.name}#{i}",
                map_status="새 라벨",
            )
            records.append(record)
    if fixed_path.is_file():
        for i, row in pd.read_csv(fixed_path).iterrows():
            record = dict.fromkeys(CORRECTION_COLS)
            stem = Path(row["file_name"]).stem
            matches = ann_df[
                (ann_df["file_name"] == row["file_name"])
                & (ann_df["category_id"] == row["category_id"])
            ]
            record.update(
                correction_id=f"kwon_fixed:{stem}:{row['category_id']}",
                stage="위치 수정",
                case=KWON_CASES.get(row["case"], row["case"]),
                file_name=row["file_name"],
                category_id=int(row["category_id"]),
                old_bbox=_as_bbox(row[["old_x", "old_y", "old_w", "old_h"]]),
                new_bbox=_as_bbox(row[BBOX_COLS]),
                method="OpenCV로 박스 없는 알약 검출 후 그 위치로 이동 + 사방 8px",
                author="gwon753-dev",
                reviewed=reviewed,
                source_record=f"{fixed_path.name}#{i}",
                map_status="확정" if len(matches) == 1 else f"후보 {len(matches)}개",
            )
            if len(matches) == 1:
                match = matches.iloc[0]
                record.update(
                    source_json=match["source_json"],
                    annotation_id=int(match["annotation_id"]),
                    input_sha256=file_sha256(ann_dir / match["source_json"]),
                )
            records.append(record)
    return pd.DataFrame(records, columns=CORRECTION_COLS)


def assign_new_ids(
    corrections: pd.DataFrame, ann_df: pd.DataFrame, start: int = NEW_ID_START
) -> pd.DataFrame:
    """annotation_id가 빈 '추가' 기록에 correction_id 순서로 새 ID를 붙인 복사본을 반환한다.

    원본 ID와 이미 기록된 ID를 피해서 붙이므로 같은 기록이면 실행마다 같은 ID가 붙는다.
    원본 ID가 start 이상이면 범위가 겹치므로 오류를 낸다.
    """
    if ann_df["annotation_id"].max() >= start:
        raise ValueError(f"원본 annotation_id가 새 ID 시작값 {start} 이상입니다.")
    corrections = corrections.copy()
    used = set(ann_df["annotation_id"]) | set(corrections["annotation_id"].dropna())
    todo = corrections[(corrections["stage"] == "추가") & corrections["annotation_id"].isna()]
    next_id = start
    for i in todo.sort_values("correction_id").index:
        while next_id in used:
            next_id += 1
        corrections.loc[i, "annotation_id"] = next_id
        used.add(next_id)
    return corrections.astype({"annotation_id": "Int64"})


def _bbox_problems(bbox: tuple[int, int, int, int], img_w: float, img_h: float) -> list[str]:
    """새 bbox가 이미지 안에 있고 크기가 양수인지 검사해 문제 목록을 반환한다."""
    x, y, w, h = bbox
    problems = []
    if w <= 0 or h <= 0:
        problems.append("새 bbox 크기 0 이하")
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        problems.append("새 bbox가 이미지 밖")
    return problems


def _new_label_row(df: pd.DataFrame, file_name: str, category_id: int) -> dict:
    """새 라벨 행의 기본값을 만든다. 사진마다 같은 열은 같은 사진에서, 약마다 같은 열은
    같은 약의 다른 라벨에서 복사한다. 어느 쪽도 아니면(예: drug_dir) 비워 둔다."""
    same_image = df[df["file_name"] == file_name]
    same_drug = df[df["category_id"] == category_id]
    row = dict.fromkeys(df.columns)
    for col in df.columns.difference(_LABEL_COLS + _ADDED_COLS):
        if not same_image.empty and same_image[col].nunique(dropna=False) <= 1:
            row[col] = same_image[col].iloc[0]
        elif not same_drug.empty and same_drug[col].nunique(dropna=False) <= 1:
            row[col] = same_drug[col].iloc[0]
    return row


def apply_corrections(
    ann_df: pd.DataFrame, corrections: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """보정 기록을 STAGES 순서로 ann_df 복사본에 적용하고 (보정본, 적용 결과)를 반환한다.

    ann_df는 바꾸지 않는다. 기록마다 아래를 모두 검사하고 하나라도 걸리면 적용하지 않는다.
    - reviewed가 True가 아니면 미검수
    - 같은 단계에 같은 대상을 가리키는 기록이 둘 이상이면 충돌(어느 값을 쓸지 먼저 정한다)
    - 고치기: 대상 라벨이 정확히 하나, segmentation 없음, 현재 bbox == old_bbox
    - 추가: 새 annotation_id가 다른 라벨과 겹치지 않고, 같은 사진에 같은 약 라벨이 없음
    - 새 bbox가 이미지 안에 있고 크기가 양수
    적용한 행은 area = w×h로 맞추고 correction_ids에 기록 ID를 남긴다. 이미 적용된 기록은
    건너뛰므로 보정본에 다시 적용해도 결과가 같다. 보정본의 label_origin은 원본/추가다.
    """
    df = ann_df.copy()
    if "correction_ids" not in df:
        df["label_origin"] = "원본"
        df["correction_ids"] = ""
    sizes = ann_df.groupby("file_name")[["width", "height"]].first()
    mapped = corrections["source_json"].notna() & (corrections["stage"] != "추가")
    targets = [
        (r.stage, r.source_json, r.annotation_id) if m else (r.stage, r.file_name, r.category_id)
        for r, m in zip(corrections.itertuples(), mapped, strict=True)
    ]
    n_same = pd.Series(targets).map(pd.Series(targets).value_counts()).to_numpy()
    logs = []
    for stage in STAGES:
        for i in corrections.index[corrections["stage"] == stage]:
            rec = corrections.loc[i]
            pos = corrections.index.get_loc(i)
            log = {col: rec.get(col) for col in APPLY_COLS}
            applied_ids = set(";".join(df["correction_ids"]).split(";"))
            if rec["correction_id"] in applied_ids:
                logs.append({**log, "status": "이미 적용", "reasons": ""})
                continue
            reviewed = pd.notna(rec["reviewed"]) and bool(rec["reviewed"])
            reasons = [] if reviewed else ["미검수"]
            if n_same[pos] > 1:
                reasons.append(f"같은 대상의 {stage} 기록 {n_same[pos]}개 (하나를 먼저 정함)")
            new_bbox = _as_bbox(rec["new_bbox"])
            if rec["file_name"] in sizes.index:
                img_w, img_h = sizes.loc[rec["file_name"]]
                reasons += _bbox_problems(new_bbox, img_w, img_h)
            else:
                reasons.append("이미지 정보 없음")
            if stage == "추가":
                same_label = (df["file_name"] == rec["file_name"]) & (
                    df["category_id"] == rec["category_id"]
                )
                if same_label.any():
                    reasons.append("같은 사진에 같은 약 라벨이 이미 있음")
                if pd.isna(rec["annotation_id"]):
                    reasons.append("새 annotation_id 없음")
                elif (df["annotation_id"] == rec["annotation_id"]).any():
                    reasons.append(f"annotation_id {rec['annotation_id']} 충돌")
            else:
                target = pd.Series(False, index=df.index)
                if rec["map_status"] != "확정":
                    reasons.append(f"대상 라벨 미확정 ({rec['map_status']})")
                else:
                    target = _label_mask(df, rec["source_json"], rec["annotation_id"])
                    if target.sum() != 1:
                        reasons.append(f"대상 라벨 {target.sum()}개")
                if target.sum() == 1:
                    row = df[target].iloc[0]
                    if row["has_segmentation"]:
                        reasons.append("segmentation 있음 (bbox만 바꾸지 않고 별도 검수)")
                    if row[BBOX_COLS].isna().any():
                        reasons.append("현재 bbox 값 결측")
                    elif _as_bbox(row[BBOX_COLS]) != _as_bbox(rec["old_bbox"]):
                        reasons.append(
                            f"현재 bbox {list(_as_bbox(row[BBOX_COLS]))} ≠ old_bbox "
                            f"{list(_as_bbox(rec['old_bbox']))} (앞 단계 보정 후 다시 계산)"
                        )
            if reasons:
                status = "미검수" if "미검수" in reasons else "거부"
                logs.append({**log, "status": status, "reasons": "; ".join(reasons)})
                continue
            x, y, w, h = new_bbox
            if stage == "추가":
                row = _new_label_row(df, rec["file_name"], rec["category_id"])
                row.update(
                    annotation_id=int(rec["annotation_id"]),
                    category_id=int(rec["category_id"]),
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    area=w * h,
                    label_origin="추가",
                    correction_ids=rec["correction_id"],
                )
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df.loc[target, BBOX_COLS] = [x, y, w, h]
                df.loc[target, "area"] = w * h
                df.loc[target, "correction_ids"] = (
                    df.loc[target, "correction_ids"] + ";" + rec["correction_id"]
                ).str.lstrip(";")
            logs.append({**log, "status": "적용", "reasons": ""})
    return df, pd.DataFrame(logs, columns=APPLY_COLS)


# JJAM 잘린 캡슐 제안(EDA 5.4). propose_bbox_at_center 결과를 값으로 고정해 OpenCV 버전과 관계없이
# 같은 라벨이 나오게 한다. 원본·제안 박스를 겹쳐 그려 캡슐 전체를 감싸는 것을 확인했다(계획 결정 2).
CUT_CAPSULE = (
    "K-003351-019232-029667_json/K-019232/K-003351-019232-029667_0_2_1_2_70_000_200.json",
    5178,
)
CUT_CAPSULE_BBOX = (527, 211, 224, 323)
CUT_CAPSULE_REVIEWED = True
# 라벨 버전마다 적용할 보정 기록의 correction_id 접두어. raw는 원본 그대로다.
# 18357은 kwon 값을 쓰므로 JJAM 색상 윤곽 제안은 기록에 넣지 않는다(계획 결정 1).
LABEL_VERSIONS = {
    "raw": (),
    "v1": ("kwon_added:", "kwon_fixed:", "jjam_capsule:"),
    "v2": ("kwon_added:", "kwon_fixed:", "jjam_capsule:", "grabcut_pad:"),
    "v2auto": ("kwon_added:", "kwon_fixed:", "jjam_capsule:", "grabcut_pad:"),
}
# 여백 보정을 검수 없이 적용하는 실험용 버전. 미검수 후보도 모두 줄이고, 거부한 행만 뺀다.
UNREVIEWED_PADDING_VERSIONS = {"v2auto"}
# 여백 보정(v2) 후보 기준: GrabCut 몸체 대비 여백 비율이 이 이상인 박스 (계획 결정 4, 10%로 조정)
PADDING_MIN = 0.10
# 사람이 채우는 여백 검수표. 최상위 csv라 팀이 검수 결과를 커밋해 공유할 수 있다.
PADDING_REVIEW_NAME = "padding_review.csv"
# decision 값. 비어 있으면 미검수이고, 승인만 v2에 적용한다.
PADDING_DECISIONS = {"승인": True, "거부": False}
PADDING_REVIEW_COLS = [
    "review_id",
    "file_name",
    "category_id",
    "dl_name",
    "source_json",
    "annotation_id",
    "old_bbox",
    "new_bbox",
    "padding_ratio",
    "bbox_iou",
    "padding_version",
    "decision",
    "reviewer",
    "comment",
]
# 라벨 버전 파일의 열. width·height는 JSON 값이 아니라 이미지 헤더에서 읽은 실제 크기다.
LABEL_FILE_COLS = [
    "file_name",
    "width",
    "height",
    "source_json",
    "annotation_id",
    "category_id",
    "dl_name",
    *BBOX_COLS,
    "area",
    "label_origin",
    "correction_ids",
]


def files_sha256(paths: Sequence[Path], root: Path) -> str:
    """파일들의 상대 경로와 내용 해시로 입력 전체의 SHA256을 만든다. 수정 시각은 보지 않는다."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(f"{path.relative_to(root).as_posix()}|{file_sha256(path)}\n".encode())
    return digest.hexdigest()


def build_corrections(
    ann_df: pd.DataFrame, ann_dir: Path, processed_dir: Path, kwon_reviewed: bool = True
) -> pd.DataFrame:
    """kwon CSV 기록, JJAM 잘린 캡슐 제안, 여백 검수표를 합쳐 새 ID를 붙인 통합 보정 기록을 만든다.

    여백 검수표(padding_review.csv)가 없으면 여백 보정 기록은 빠진다.
    """
    kwon = corrections_from_kwon(
        processed_dir / "added_annotations.csv",
        processed_dir / "fixed_annotations.csv",
        ann_df,
        ann_dir,
        reviewed=kwon_reviewed,
    )
    capsule = correction_for_label(
        ann_df,
        *CUT_CAPSULE,
        CUT_CAPSULE_BBOX,
        ann_dir,
        correction_id=f"jjam_capsule:{CUT_CAPSULE[0]}#{CUT_CAPSULE[1]}",
        stage="위치 수정",
        case="잘린 bbox",
        author="potato",
        reviewed=CUT_CAPSULE_REVIEWED,
        method="박스 중앙의 HSV 색상 덩어리 + 사방 8px",
        source_record="JJAM99_eda.ipynb 셀 48 (EDA 5.4에서 다시 계산한 값)",
    )
    corrections = assign_new_ids(
        pd.concat([kwon, pd.DataFrame([capsule])], ignore_index=True), ann_df
    )
    # 여백 기록은 추가 라벨을 새 ID로 가리키므로, 새 ID를 붙인 뒤에 합쳐야 ID가 v1과 같게 유지된다.
    review_path = processed_dir / PADDING_REVIEW_NAME
    if review_path.is_file():
        padding = padding_corrections(load_padding_review(review_path), ann_dir)
        corrections = pd.concat([corrections, padding], ignore_index=True)
    return corrections.astype({"annotation_id": "Int64"})


def save_corrections(corrections: pd.DataFrame, path: Path) -> None:
    """보정 기록을 CSV로 저장한다. bbox 튜플은 "[x, y, w, h]" 문자열로 적는다."""
    out = corrections.copy()
    for col in ["old_bbox", "new_bbox"]:
        out[col] = out[col].map(lambda b: list(b) if isinstance(b, tuple) else b)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")


def make_label_version(
    ann_df: pd.DataFrame, corrections: pd.DataFrame, version: str, image_sizes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LABEL_VERSIONS의 기록만 적용해 (라벨 표, 적용 결과)를 반환한다.

    image_sizes는 file_name, width, height 열의 표(read_image_headers 결과)이며 라벨의
    width·height를 이 값으로 바꾼다. 승인되지 않은 여백 보정은 고르지 않고, 고른 기록 중
    적용되지 않은 것이 있으면 오류를 낸다.
    """
    # 접두어가 없으면(raw) startswith(())가 모두 False라 기록을 하나도 고르지 않는다.
    selected = corrections[corrections["correction_id"].str.startswith(LABEL_VERSIONS[version])]
    # 여백 보정은 검수표에서 승인한 것만 쓴다. 미검수·거부는 오류가 아니라 적용 대상이 아니다.
    padding = selected["stage"] == "여백 보정"
    if version in UNREVIEWED_PADDING_VERSIONS:
        # 거부한 행만 빼고, 미검수 행은 승인한 것처럼 적용한다. corrections.csv에는 미검수로 남는다.
        selected = selected[~(padding & selected["reviewed"].eq(False))].copy()
        selected.loc[selected["stage"] == "여백 보정", "reviewed"] = True
    else:
        selected = selected[~(padding & ~selected["reviewed"].eq(True))]
    labels, log = apply_corrections(ann_df, selected)
    failed = log[log["status"] != "적용"]
    if not failed.empty:
        raise ValueError(f"{version}에서 적용되지 않은 보정 기록:\n{failed.to_string()}")
    sizes = image_sizes.set_index("file_name")
    labels["width"] = labels["file_name"].map(sizes["width"])
    labels["height"] = labels["file_name"].map(sizes["height"])
    labels = labels[LABEL_FILE_COLS].sort_values(["file_name", "category_id", "annotation_id"])
    return labels.reset_index(drop=True), log


def check_labels(labels: pd.DataFrame) -> dict[str, int]:
    """라벨 표의 bbox 문제 수를 센다. area_mismatch를 뺀 값이 모두 0이어야 학습에 쓸 수 있다.

    area는 학습에 쓰지 않으므로 area_mismatch(JSON area ≠ w×h)는 기록만 한다.
    """
    x, y, w, h = (labels[col] for col in BBOX_COLS)
    return {
        "n_labels": len(labels),
        "n_images": labels["file_name"].nunique(),
        "n_classes": labels["category_id"].nunique(),
        "bbox_missing": int(labels[BBOX_COLS].isna().any(axis=1).sum()),
        "out_of_image": int(
            ((x < 0) | (y < 0) | (x + w > labels["width"]) | (y + h > labels["height"])).sum()
        ),
        "non_positive": int(((w <= 0) | (h <= 0)).sum()),
        "duplicate_label": int(labels.duplicated(["file_name", "category_id"]).sum()),
        "area_mismatch": int((labels["area"] != w * h).sum()),
    }


def to_yolo_lines(boxes: pd.DataFrame, class_idx: dict[int, int]) -> tuple[list[str], int]:
    """한 사진의 라벨을 YOLO 줄("cls cx cy w h", 0–1 정규화)로 바꿔 (줄, 버린 박스 수)를 반환한다.

    이미지 밖으로 나간 부분은 잘라 내고, 잘라서 크기가 0 이하가 된 박스는 버린다(raw 버전용).
    """
    lines, n_dropped = [], 0
    for row in boxes.itertuples():
        x1, y1 = max(row.x, 0), max(row.y, 0)
        x2, y2 = min(row.x + row.w, row.width), min(row.y + row.h, row.height)
        if x2 <= x1 or y2 <= y1:
            n_dropped += 1
            continue
        cx, cy = (x1 + x2) / 2 / row.width, (y1 + y2) / 2 / row.height
        w, h = (x2 - x1) / row.width, (y2 - y1) / row.height
        lines.append(f"{class_idx[row.category_id]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines, n_dropped


def write_yolo_labels(
    labels: pd.DataFrame, file_names: Sequence[str], class_idx: dict[int, int], out_dir: Path
) -> dict[str, int]:
    """사진 1장 = txt 1개로 YOLO 라벨을 쓰고 변환 결과 수를 반환한다.

    out_dir의 기존 txt와 ultralytics 캐시(*.cache)는 지운다. 캐시는 파일 크기로만 바뀜을 알아채므로
    좌표만 바뀌면 옛 라벨을 쓰기 때문이다. 라벨이 없는 사진은 빈 txt를 만든다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in [*out_dir.glob("*.txt"), *out_dir.parent.glob("*.cache")]:
        old.unlink()
    by_image = dict(tuple(labels.groupby("file_name")))
    n_lines = n_dropped = n_empty = 0
    for file_name in file_names:
        boxes = by_image.get(file_name, labels.iloc[0:0])
        lines, dropped = to_yolo_lines(boxes, class_idx)
        (out_dir / Path(file_name).with_suffix(".txt").name).write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8"
        )
        n_lines += len(lines)
        n_dropped += dropped
        n_empty += not lines
    return {
        "n_txt": len(file_names),
        "n_lines": n_lines,
        "n_dropped": n_dropped,
        "n_empty": n_empty,
    }


def read_yolo_labels(txt_path: Path, width: float, height: float) -> pd.DataFrame:
    """YOLO txt를 읽어 yolo_idx와 픽셀 단위 COCO bbox(x, y, w, h) 표로 되돌린다."""
    rows = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        cls, cx, cy, w, h = line.split()
        w_px, h_px = float(w) * width, float(h) * height
        x_px, y_px = float(cx) * width - w_px / 2, float(cy) * height - h_px / 2
        rows.append({"yolo_idx": int(cls), "x": x_px, "y": y_px, "w": w_px, "h": h_px})
    return pd.DataFrame(rows, columns=["yolo_idx", *BBOX_COLS])


def _bbox_text(values: Sequence[float]) -> str:
    return json.dumps([int(v) for v in values])


def padding_review_table(
    padding_df: pd.DataFrame, labels: pd.DataFrame, padding_min: float = PADDING_MIN
) -> pd.DataFrame:
    """measure_bbox_padding 결과에서 여백 보정 후보만 골라 검수표를 만든다.

    후보: status가 estimated이고 padding_ratio ≥ padding_min. 새 bbox는 GrabCut 몸체 + 경계 여유
    (tight_bbox) 값을 그대로 고정해 둔다. decision·reviewer·comment는 사람이 채운다.
    """
    ok = (padding_df["status"] == "estimated") & (padding_df["padding_ratio"] >= padding_min)
    cand = padding_df[ok].sort_values(["file_name", "category_id"]).reset_index(drop=True)
    names = labels.drop_duplicates("category_id").set_index("category_id")["dl_name"]
    review = pd.DataFrame(
        {
            "review_id": [f"P{i + 1:03d}" for i in range(len(cand))],
            "file_name": cand["file_name"],
            "category_id": cand["category_id"].astype(int),
            "dl_name": cand["category_id"].map(names),
            "source_json": cand["source_json"],
            "annotation_id": cand["annotation_id"].astype(int),
            "old_bbox": cand["bbox"].map(_bbox_text),
            "new_bbox": cand["tight_bbox"].map(_bbox_text),
            "padding_ratio": cand["padding_ratio"].round(4),
            "bbox_iou": cand["bbox_iou"].round(4),
            "padding_version": PADDING_VERSION,
            "decision": "",
            "reviewer": "",
            "comment": "",
        }
    )
    return review[PADDING_REVIEW_COLS]


def merge_padding_review(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """이전 검수표의 decision·reviewer·comment를 같은 라벨·같은 old/new bbox 행에 옮긴다.

    bbox 값이 바뀐 행은 다시 검수해야 하므로 옮기지 않는다.
    """
    keys = ["source_json", "annotation_id", "old_bbox", "new_bbox"]
    kept = old[[*keys, "decision", "reviewer", "comment"]].fillna({"source_json": ""})
    merged = new.drop(columns=["decision", "reviewer", "comment"]).fillna({"source_json": ""})
    merged = merged.merge(kept, on=keys, how="left")
    merged["source_json"] = merged["source_json"].replace("", None)
    return merged.fillna({"decision": "", "reviewer": "", "comment": ""})[PADDING_REVIEW_COLS]


def load_padding_review(path: Path) -> pd.DataFrame:
    """여백 검수표를 읽는다. decision이 승인/거부/빈칸이 아니면 오류를 낸다."""
    review = pd.read_csv(path, dtype={"decision": str, "reviewer": str, "comment": str})
    review[["decision", "reviewer", "comment"]] = review[
        ["decision", "reviewer", "comment"]
    ].fillna("")
    review["decision"] = review["decision"].str.strip()
    bad = review[~review["decision"].isin(["", *PADDING_DECISIONS])]
    if not bad.empty:
        raise ValueError(f"decision은 승인/거부/빈칸만 씁니다:\n{bad[['review_id', 'decision']]}")
    return review


def padding_corrections(review: pd.DataFrame, ann_dir: Path) -> pd.DataFrame:
    """여백 검수표를 '여백 보정' 단계의 보정 기록으로 바꾼다. 승인한 행만 reviewed=True다.

    old_bbox는 v1 라벨에서 잰 값이라 v1 보정 뒤의 bbox와 같을 때만 적용된다.
    """
    records = []
    for row in review.itertuples():
        source_json = None if pd.isna(row.source_json) else row.source_json
        record = dict.fromkeys(CORRECTION_COLS)
        record.update(
            correction_id=f"grabcut_pad:{row.file_name}#{row.category_id}",
            stage="여백 보정",
            case="bbox 여백",
            source_json=source_json,
            file_name=row.file_name,
            category_id=int(row.category_id),
            annotation_id=int(row.annotation_id),
            old_bbox=_as_bbox(json.loads(row.old_bbox)),
            new_bbox=_as_bbox(json.loads(row.new_bbox)),
            method=f"GrabCut 몸체 + 경계 여유 ({row.padding_version})",
            author=row.reviewer or None,
            reviewed=PADDING_DECISIONS.get(row.decision),
            source_record=f"{PADDING_REVIEW_NAME}#{row.review_id}",
            input_sha256=None if source_json is None else file_sha256(ann_dir / source_json),
            map_status="확정",
        )
        records.append(record)
    return pd.DataFrame(records, columns=CORRECTION_COLS)
