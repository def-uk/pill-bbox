import hashlib
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from data_process.data_processing import BBOX_COLS

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
    """원본 라벨 하나의 bbox를 고치는 보정 기록 한 행을 만든다. old_bbox는 ann_df의 현재 값이다.

    fields에는 correction_id, stage, case, method, author, reviewed, source_record를 넘긴다.
    """
    is_target = (ann_df["source_json"] == source_json) & (ann_df["annotation_id"] == annotation_id)
    target = ann_df[is_target]
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
        input_sha256=file_sha256(ann_dir / source_json),
        map_status="확정",
    )
    return record


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
                    target = (df["source_json"] == rec["source_json"]) & (
                        df["annotation_id"] == rec["annotation_id"]
                    )
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
