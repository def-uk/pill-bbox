import hashlib
import itertools
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg"}
BBOX_COLS = ["x", "y", "w", "h"]

# 이미지 단위로 같은 값을 갖는 촬영 정보. 나머지 필드(drug_N, dl_name 등)는 알약마다 다르다.
IMAGE_COLS = [
    "file_name",
    "width",
    "height",
    "back_color",
    "light_color",
    "camera_la",
    "camera_lo",
    "size",
]

# 오류 표의 열. excluded는 그 입력이 ann_df에서 빠졌는지 나타낸다.
ERROR_COLS = ["source_json", "annotation_id", "stage", "message", "excluded"]


def list_images(folder: Path) -> list[Path]:
    """Zone.Identifier 등을 빼고 이미지 파일만 반환한다. 숫자 이름은 숫자 순서로 정렬한다."""
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(
        paths, key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else 0, p.name)
    )


def _as_objects(value: object) -> list[dict]:
    """JSON의 단일 객체와 객체 목록을 같은 형태로 다룬다."""
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ValueError(f"객체 또는 객체 목록이 필요합니다: {type(value).__name__}")


def _parse_bbox(bbox: object) -> tuple[float, float, float, float]:
    """bbox를 (x, y, w, h)로 바꾼다. 값이 없거나 형식이 다르면 NaN 4개를 반환한다."""
    try:
        x, y, w, h = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return (float("nan"),) * 4
    return x, y, w, h


def _annotation_row(
    ann: dict, images: list[dict], categories: list[dict], source_json: str
) -> tuple[dict | None, list[dict]]:
    """annotation 하나를 ann_df의 한 행으로 만든다. 연결에 실패하면 행 대신 오류를 돌려준다."""
    ann_id = ann.get("id")
    errors = []
    matches = [img for img in images if img.get("id") == ann.get("image_id")]
    if len(matches) != 1:
        message = f"JSON 내부 image_id={ann.get('image_id')} 연결 대상이 {len(matches)}개"
        return None, [_error(source_json, ann_id, "image_link", message, excluded=True)]
    img = {k: v for k, v in matches[0].items() if k != "id"}

    cat_matches = [c for c in categories if c.get("id") == ann.get("category_id")]
    if len(cat_matches) != 1:
        message = f"JSON 내부 category_id={ann.get('category_id')} 연결 대상이 {len(cat_matches)}개"
        errors.append(_error(source_json, ann_id, "category_link", message, excluded=False))
    category_name = cat_matches[0].get("name") if len(cat_matches) == 1 else None

    x, y, w, h = _parse_bbox(ann.get("bbox"))
    row = {
        **img,
        "image_id": ann.get("image_id"),
        "source_json": source_json,
        "annotation_id": ann_id,
        "category_id": ann.get("category_id"),
        "category_name": category_name,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area": ann.get("area"),
        "iscrowd": ann.get("iscrowd"),
        "ignore": ann.get("ignore"),
        "has_segmentation": bool(ann.get("segmentation")),
    }
    return row, errors


def _error(
    source_json: str, ann_id: object, stage: str, message: str, excluded: bool
) -> dict[str, object]:
    """오류 표의 한 행을 만든다."""
    return dict(zip(ERROR_COLS, [source_json, ann_id, stage, message, excluded], strict=True))


def load_annotations(ann_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """annotation JSON을 모두 읽어 (원본 bbox당 한 행인 ann_df, 오류 표)를 반환한다.

    이미지 메타데이터는 JSON 안의 annotation.image_id로 연결한다(전역 image_id로 합치지 않음).
    source_json은 ann_dir 기준 상대 경로이며, (source_json, annotation_id)가 라벨 식별자다.
    bbox 값이 비어 있거나 형식이 다른 라벨은 빼지 않고 x, y, w, h를 NaN으로 둔다.
    """
    rows, errors = [], []
    for json_path in sorted(ann_dir.rglob("*.json")):
        source_json = json_path.relative_to(ann_dir).as_posix()
        try:
            data = json.loads(json_path.read_text(encoding="utf-8-sig"))
            images = _as_objects(data["images"])
            annotations = _as_objects(data["annotations"])
            categories = _as_objects(data.get("categories", []))
        except (OSError, UnicodeDecodeError, ValueError, KeyError) as exc:
            errors.append(_error(source_json, None, "parse", repr(exc), excluded=True))
            continue
        if not annotations:
            errors.append(_error(source_json, None, "empty", "annotation 없음", excluded=False))
        for ann in annotations:
            row, row_errors = _annotation_row(ann, images, categories, source_json)
            errors.extend(row_errors)
            if row is not None:
                rows.append(row)

    ann_df = pd.DataFrame(rows)
    # 오류가 없어도 excluded를 bool로 두어야 error_df[error_df["excluded"]]가 행 선택이 된다.
    error_df = pd.DataFrame(errors, columns=ERROR_COLS).astype({"excluded": bool})
    return ann_df, error_df


def build_img_df(ann_df: pd.DataFrame, img_paths: list[Path]) -> pd.DataFrame:
    """디스크의 train 이미지 목록에서 시작해 이미지 한 장당 한 행인 표를 만든다.

    라벨이 없는 이미지도 포함한다(n_boxes=0). 촬영 정보는 같은 이미지의 첫 JSON 값을 쓴다.
    combo는 파일 이름 첫 밑줄 앞의 알약 조합, n_codes_in_name은 이름에 적힌 약 코드 수다.
    """
    img_df = pd.DataFrame({"file_name": [p.name for p in img_paths]})
    img_meta = ann_df.sort_values("source_json")[IMAGE_COLS].drop_duplicates("file_name")
    img_df = img_df.merge(img_meta, on="file_name", how="left")
    img_df["n_boxes"] = img_df["file_name"].map(ann_df["file_name"].value_counts())
    img_df["n_json"] = img_df["file_name"].map(ann_df.groupby("file_name")["source_json"].nunique())
    img_df[["n_boxes", "n_json"]] = img_df[["n_boxes", "n_json"]].fillna(0).astype(int)
    img_df["combo"] = img_df["file_name"].str.split("_").str[0]
    img_df["n_codes_in_name"] = img_df["combo"].str.count("-")
    return img_df


def build_cat_df(ann_df: pd.DataFrame) -> pd.DataFrame:
    """category_id ↔ 약 이름 표를 만든다. bbox 수와 이미지 수를 함께 둔다."""
    first = ann_df.sort_values("source_json").drop_duplicates("category_id")
    cat_df = first[["category_id", "category_name", "drug_N", "dl_name", "dl_name_en"]]
    counts = ann_df.groupby("category_id").agg(
        n_boxes=("annotation_id", "size"), n_images=("file_name", "nunique")
    )
    return cat_df.merge(counts, on="category_id").sort_values("category_id").reset_index(drop=True)


def load_schema(readme_path: Path) -> pd.DataFrame:
    """data/README.md의 annotation 스키마 표를 읽어 키별 타입·필수여부·설명 표를 반환한다.

    필수여부: M = 필수, O = 선택, C = 조건부. 예: | `images[].drug_shape` | String | M | 알약 모양 |
    """
    pattern = re.compile(r"^\| `(\w+)(?:\[\]\.(\w+))?` \| ([^|]+?) \| (\w) \| ([^|]+?) \|$")
    rows = []
    for line in readme_path.read_text(encoding="utf-8").splitlines():
        if m := pattern.match(line.strip()):
            section, field, dtype, required, description = m.groups()
            rows.append([section, field or "", dtype, required, description])
    return pd.DataFrame(rows, columns=["section", "field", "type", "required", "description"])


def _image_error(exc: Exception, path: Path) -> str:
    """오류 메시지의 절대 경로를 파일 이름으로 바꿔 출력과 캐시에 개인 경로가 남지 않게 한다."""
    return repr(exc).replace(str(path), path.name)


def read_image_headers(paths: list[Path], split: str) -> pd.DataFrame:
    """이미지 헤더만 읽어 크기·색상 모드를 표로 만든다. 픽셀은 디코딩하지 않는다."""
    rows = []
    for path in paths:
        row = {"split": split, "file_name": path.name, "width": None, "height": None}
        row.update(mode=None, error=None)
        try:
            with Image.open(path) as img:
                row.update(width=img.width, height=img.height, mode=img.mode)
        except (OSError, ValueError) as exc:
            row["error"] = _image_error(exc, path)
        rows.append(row)
    return pd.DataFrame(rows, columns=["split", "file_name", "width", "height", "mode", "error"])


def inspect_image_integrity(paths: list[Path], split: str) -> pd.DataFrame:
    """이미지를 한 장씩 전체 디코딩해 파일 SHA256·픽셀 SHA256·디코딩 오류를 표로 만든다.

    file_sha256은 파일 바이트, pixel_sha256은 RGB로 바꾼 픽셀의 해시다. 파일 형식이 달라도
    픽셀이 같으면 pixel_sha256이 같다. 이미지 수에 비례해 오래 걸린다.
    """
    cols = ["split", "file_name", "file_sha256", "pixel_sha256", "decoded", "error"]
    rows = []
    for path in paths:
        row = dict.fromkeys(cols)
        row.update(split=split, file_name=path.name, decoded=False)
        try:
            with path.open("rb") as stream:
                row["file_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            # verify()는 형식만 검사하므로 다시 열어 전체 픽셀을 디코딩한다.
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                rgb = img.convert("RGB")
            row["pixel_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
            row["decoded"] = True
        except (OSError, ValueError, SyntaxError) as exc:
            row["error"] = _image_error(exc, path)
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def fingerprint_files(paths: list[Path], root: Path) -> str:
    """파일 목록의 상대 경로·크기·수정 시각으로 입력 지문(SHA256)을 만든다. 캐시 검증용."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        digest.update(
            f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def load_cache(cache_path: Path, key: dict) -> tuple[pd.DataFrame | None, str]:
    """key가 저장 당시와 모두 같을 때만 캐시 표를 반환한다. 아니면 (None, 이유)를 반환한다."""
    if not cache_path.is_file():
        return None, "캐시 없음"
    try:
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        saved_key, records = saved["key"], saved["records"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"캐시 읽기 실패: {exc!r}"
    diff = sorted(k for k in key.keys() | saved_key.keys() if key.get(k) != saved_key.get(k))
    if diff:
        return None, f"캐시 불일치: {', '.join(diff)}"
    return pd.DataFrame(records), "캐시 사용"


def save_cache(cache_path: Path, key: dict, df: pd.DataFrame) -> None:
    """표를 key와 함께 JSON으로 저장한다. load_cache는 key가 같을 때만 이 표를 돌려준다."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(df.to_json(orient="records", force_ascii=False))
    payload = {"key": key, "records": records}
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _missing_kind(value: object, present: bool) -> str:
    """JSON 값 하나의 결측 유형을 돌려준다. 0과 False는 결측으로 보지 않는다."""
    if not present:
        return "키 없음"
    if value is None:
        return "null"
    if isinstance(value, str) and not value.strip():
        return "빈 문자열"
    if isinstance(value, (list, dict)) and not value:
        return "빈 배열/객체"
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN/무한대"
    return "값 있음"


def scan_missing_values(ann_dir: Path, schema_df: pd.DataFrame) -> pd.DataFrame:
    """원본 JSON을 다시 읽어 스키마의 모든 키를 레코드마다 검사하고 결측 유형을 기록한다.

    로더는 키 없음과 null을 모두 NaN으로 만들므로, 둘을 구분하려면 원본을 읽어야 한다.
    한 행 = (JSON, 섹션, 레코드 인덱스, 키). kind는 _missing_kind의 값이며 결측이 아니면
    "값 있음"이다. 읽지 못한 JSON은 로더의 오류 표(parse)에 이미 있으므로 건너뛴다.
    """
    fields = list(schema_df[["section", "field", "required"]].itertuples(index=False))
    rows = []
    for json_path in sorted(ann_dir.rglob("*.json")):
        source_json = json_path.relative_to(ann_dir).as_posix()
        try:
            data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for section, field, required in fields:
            if not field:
                # 최상위 키(images, annotations 등)는 JSON 자체를 레코드 하나로 본다.
                records, key_name = [data], section
            else:
                value = data.get(section)
                records = value if isinstance(value, list) else [value] if value else []
                key_name = field
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                kind = _missing_kind(record.get(key_name), key_name in record)
                if field == "bbox" and kind == "값 있음" and isinstance(record[key_name], list):
                    parts = [_missing_kind(v, True) for v in record[key_name]]
                    if any(p != "값 있음" for p in parts):
                        kind = "bbox 내부 결측"
                rows.append(
                    {
                        "source_json": source_json,
                        "section": section,
                        "record_index": index,
                        "key": f"{section}.{field}" if field else section,
                        "required": required,
                        "kind": kind,
                        "file_name": record.get("file_name") if section == "images" else None,
                        "drug_N": record.get("drug_N") if section == "images" else None,
                    }
                )
    return pd.DataFrame(rows)


def compare_name_codes(img_df: pd.DataFrame, ann_df: pd.DataFrame) -> pd.DataFrame:
    """사진마다 파일명의 약 코드(K-003351 → 3351)와 라벨된 category_id를 비교한다.

    missing_codes는 파일명에는 있는데 bbox가 없는 약, extra_codes는 파일명에 없는데 bbox가
    있는 약이다. n_boxes는 넘겨받은 ann_df 기준으로 다시 센다(보정본을 넘기면 보정본 기준).
    """
    name_codes = img_df["combo"].str.split("-").str[1:].map(lambda codes: {int(c) for c in codes})
    labeled_by_file = ann_df.groupby("file_name")["category_id"].apply(set)
    labeled = img_df["file_name"].map(lambda name: labeled_by_file.get(name, set()))
    pairs = list(zip(name_codes, labeled, strict=True))
    n_boxes = img_df["file_name"].map(ann_df["file_name"].value_counts()).fillna(0).astype(int)
    return img_df[["file_name", "combo", "n_codes_in_name"]].assign(
        n_boxes=n_boxes,
        missing_codes=[sorted(name - lab) for name, lab in pairs],
        extra_codes=[sorted(lab - name) for name, lab in pairs],
    )


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """두 [x, y, w, h] 박스의 IoU(겹친 넓이 ÷ 합친 넓이)를 반환한다."""
    inter_w = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def bbox_gap(a: Sequence[float], b: Sequence[float]) -> float:
    """두 [x, y, w, h] 박스 사이의 최소 간격(px)을 반환한다. 맞닿거나 겹치면 0이다."""
    dx = max(b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]), 0)
    dy = max(b[1] - (a[1] + a[3]), a[1] - (b[1] + b[3]), 0)
    return math.hypot(dx, dy)


def bbox_flags(df: pd.DataFrame, area_tol: float = 1.0) -> pd.DataFrame:
    """bbox마다 좌표 오류 여부를 표로 만든다. 열: out_of_image, non_positive, area_mismatch.

    out_of_image는 박스 일부라도 이미지(width × height) 밖에 있는 경우, area_mismatch는
    |area − w×h| > area_tol인 경우다. 좌표가 NaN인 라벨(4.1)은 세 검사 모두 False로 둔다.
    """
    has_box = df[BBOX_COLS].notna().all(axis=1)
    out_of_image = (
        (df["x"] < 0)
        | (df["y"] < 0)
        | (df["x"] + df["w"] > df["width"])
        | (df["y"] + df["h"] > df["height"])
    )
    non_positive = (df["w"] <= 0) | (df["h"] <= 0)
    area_diff = (pd.to_numeric(df["area"], errors="coerce") - df["w"] * df["h"]).abs()
    return pd.DataFrame(
        {
            "out_of_image": out_of_image & has_box,
            "non_positive": non_positive & has_box,
            "area_mismatch": (area_diff > area_tol) & has_box,
        },
        index=df.index,
    )


def find_box_pairs(ann_df: pd.DataFrame, max_gap: float = 10) -> pd.DataFrame:
    """같은 사진의 bbox 쌍 중 간격이 max_gap(px) 이하인 쌍을 간격·IoU와 함께 반환한다.

    한 행 = bbox 쌍. _1, _2 열은 두 라벨의 (source_json, annotation_id, category_id)다.
    IoU가 1에 가까우면 같은 좌표를 두 약에 붙인 것이고, 작으면 알약끼리 맞닿은 것이다.
    """
    keys = ["source_json", "annotation_id", "category_id"]
    cols = ["file_name", *[f"{k}_{n}" for n in (1, 2) for k in keys], "gap_px", "iou"]
    rows = []
    for file_name, group in ann_df.dropna(subset=BBOX_COLS).groupby("file_name"):
        labels = group[[*keys, *BBOX_COLS]].itertuples(index=False)
        for a, b in itertools.combinations(labels, 2):
            box_a, box_b = (a.x, a.y, a.w, a.h), (b.x, b.y, b.w, b.h)
            gap = bbox_gap(box_a, box_b)
            if gap <= max_gap:
                ids = [getattr(label, k) for label in (a, b) for k in keys]
                rows.append([file_name, *ids, round(gap, 1), bbox_iou(box_a, box_b)])
    return pd.DataFrame(rows, columns=cols)


def iqr_bounds(values: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """IQR 기준 (하한, 상한)을 반환한다. 범위 밖은 눈으로 확인할 후보이지 오류 판정이 아니다."""
    q1, q3 = values.quantile([0.25, 0.75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


def _clip_box(bbox: Sequence[float], img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """bbox를 이미지 안으로 잘라 (x0, y0, x1, y1) 정수 좌표로 바꾼다. 겹치는 부분이 없으면 None."""
    x0, y0 = max(int(bbox[0]), 0), max(int(bbox[1]), 0)
    x1, y1 = min(int(bbox[0] + bbox[2]), img_w), min(int(bbox[1] + bbox[3]), img_h)
    return (x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None


def _background_diff(img_bgr: np.ndarray) -> np.ndarray:
    """픽셀마다 배경 추정값과의 LAB 차이 벡터를 구한다(kwon 방식).

    1/8로 줄인 사진에 큰 median blur(원본 기준 약 500px)를 걸어 배경을 추정한다. 창이 커야
    큰 알약 안쪽까지 배경으로 착각하지 않는다.
    """
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    small = cv2.resize(lab, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, 61), (w, h), interpolation=cv2.INTER_LINEAR)
    return lab.astype(np.float32) - bg.astype(np.float32)


def _pill_mask(diff: np.ndarray, thresh: float = 25) -> np.ndarray:
    """배경과 색이 다른 픽셀(알약)을 255로 표시한다. 어둡기만 하고 색이 같은 그림자는 뺀다."""
    shadow = (diff[..., 0] < 0) & (np.linalg.norm(diff[..., 1:], axis=2) < 10)
    mask = ((np.linalg.norm(diff, axis=2) > thresh) & ~shadow).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((21, 21), np.uint8))


def measure_box_pixels(
    ann_df: pd.DataFrame, img_dir: Path, min_blob_area: int = 500
) -> pd.DataFrame:
    """bbox 안의 픽셀을 배경과 비교해 빈 배경·모양 이상 지표를 잰다(kwon 셀 62, 70).

    ann_df와 같은 index의 표를 반환한다. 사진은 한 장씩 한 번만 읽는다.
    - diff_p90: 박스 안 픽셀의 배경 대비 색 차이 상위 10% 값. 빈 배경이면 0에 가깝다.
    - fill: 박스 넓이 중 알약 픽셀 비율. 투명·어두운 알약은 알약이 있어도 낮게 나온다.
    - pill_area: 박스 안 알약 픽셀 수. 박스와 달리 알약이 회전해도 거의 변하지 않는다.
    - solidity: 가장 큰 윤곽 넓이 ÷ 볼록 껍질 넓이. 깨지거나 두 알이 붙으면 낮아진다.
    - n_blobs: 넓이 min_blob_area 이상인 덩어리 수.
    박스가 이미지와 겹치지 않거나 사진을 읽지 못하면 지표를 NaN으로 두고 status에 이유를 쓴다.
    """
    metrics = ["diff_p90", "fill", "pill_area", "solidity", "n_blobs"]
    rows = {}
    for file_name, group in ann_df.dropna(subset=BBOX_COLS).groupby("file_name"):
        img = cv2.imread(str(img_dir / file_name))
        if img is None:
            rows.update({i: {"status": "사진 없음"} for i in group.index})
            continue
        diff = _background_diff(img)
        dist, mask = np.linalg.norm(diff, axis=2), _pill_mask(diff)
        for i, box in group[BBOX_COLS].iterrows():
            clipped = _clip_box(box.to_numpy(), img.shape[1], img.shape[0])
            if clipped is None:
                rows[i] = {"status": "이미지 밖"}
                continue
            x0, y0, x1, y1 = clipped
            crop = mask[y0:y1, x0:x1]
            contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            main = max(contours, key=cv2.contourArea) if contours else None
            hull_area = cv2.contourArea(cv2.convexHull(main)) if main is not None else 0
            rows[i] = {
                "status": "측정",
                "diff_p90": float(np.percentile(dist[y0:y1, x0:x1], 90)),
                "fill": float((crop > 0).sum() / (box["w"] * box["h"])),
                "pill_area": int((crop > 0).sum()),
                "solidity": cv2.contourArea(main) / hull_area if hull_area else 0.0,
                "n_blobs": sum(cv2.contourArea(c) >= min_blob_area for c in contours),
            }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["status", *metrics]).reindex(
        ann_df.index
    )


# 배경과 갈색·노란 캡슐을 나누는 HSV 범위(JJAM 셀 26, 48). 특정 사진에 맞춘 값이다.
HSV_PILL_RANGE = ((0, 45, 35), (45, 255, 255))


def _color_mask(rgb: np.ndarray) -> np.ndarray:
    """HSV_PILL_RANGE 안의 픽셀을 255로 표시하고 작은 구멍과 점을 정리한다."""
    lower, upper = HSV_PILL_RANGE
    mask = cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), np.array(lower), np.array(upper))
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _pad_box(rect: Sequence[int], padding: int, img_w: int, img_h: int) -> list[int]:
    """[x, y, w, h]에 사방 padding px를 더하고 이미지 안으로 자른다."""
    x0, y0 = max(0, rect[0] - padding), max(0, rect[1] - padding)
    x1 = min(img_w, rect[0] + rect[2] + padding)
    y1 = min(img_h, rect[1] + rect[3] + padding)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def propose_bbox_by_shape(
    img_path: Path, old_bbox: Sequence[float], padding: int = 8, min_margin: float = 0.05
) -> tuple[list[int] | None, str]:
    """x 좌표를 믿을 수 없는 bbox의 새 위치를 색상 윤곽에서 찾는다(JJAM 셀 26).

    윤곽 중 y·너비·높이가 old_bbox와 가장 비슷한 것에 사방 padding px를 더한다. 기존 y·크기가
    맞다는 가정이 있고, 1·2위 점수 차가 min_margin보다 작으면 대상을 정할 수 없어 None을 반환한다.
    반환값은 (제안 bbox 또는 None, 설명)이다.
    """
    with Image.open(img_path) as img:
        rgb = np.array(img.convert("RGB"))
    img_h, img_w = rgb.shape[:2]
    _, old_y, old_w, old_h = old_bbox
    contours, _ = cv2.findContours(_color_mask(rgb), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        if not 2000 < cv2.contourArea(contour) < img_w * img_h * 0.3:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        score = abs(y - old_y) / img_h + abs(w - old_w) / img_w + abs(h - old_h) / img_h
        candidates.append((score, [x, y, w, h]))
    candidates.sort(key=lambda c: c[0])
    if not candidates:
        return None, "윤곽 후보 없음"
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < min_margin:
        return None, f"1·2위 점수 차 {candidates[1][0] - candidates[0][0]:.3f} < {min_margin}"
    return _pad_box(candidates[0][1], padding, img_w, img_h), f"점수 {candidates[0][0]:.3f}"


def propose_bbox_at_center(
    img_path: Path, old_bbox: Sequence[float], padding: int = 8
) -> tuple[list[int] | None, str]:
    """old_bbox 중앙에 있는 색상 덩어리 전체를 감싸는 bbox를 제안한다(JJAM 셀 48, 잘린 캡슐).

    반환값은 (제안 bbox 또는 None, 설명)이다. 중앙이 덩어리에 속하지 않으면 None이다.
    """
    with Image.open(img_path) as img:
        rgb = np.array(img.convert("RGB"))
    img_h, img_w = rgb.shape[:2]
    _, components, stats, _ = cv2.connectedComponentsWithStats(_color_mask(rgb))
    x, y, w, h = (int(v) for v in old_bbox)
    component_id = int(components[y + h // 2, x + w // 2])
    if component_id == 0:
        return None, "박스 중앙에 색상 덩어리 없음"
    left, top, width, height, n_pixels = (int(v) for v in stats[component_id])
    return _pad_box([left, top, width, height], padding, img_w, img_h), f"덩어리 {n_pixels}px"


# GrabCut 여백 추정 방식의 이름(JJAM 셀 54). 계산 방식이나 기본 인자를 바꾸면 이름도 바꾼다.
PADDING_VERSION = "shadow_chroma_reference_v3"


def _mark_shadow_background(
    crop: np.ndarray,
    mask: np.ndarray,
    rectangle: tuple[int, int, int, int],
    score_threshold: float = 0.25,
    darkness_ratio: float = 0.95,
) -> None:
    """배경과 색 비율이 비슷한 어두운 부분을 배경으로 지정하고 알약 끝은 보호한다."""
    # 밝기 변화에 비교적 안정적인 파랑/초록·빨강/초록 색 비율을 사용한다.
    colors = crop.astype(np.float32) + 5.0
    chromaticity = np.log(colors[:, :, [0, 2]] / colors[:, :, [1]])
    background = chromaticity[mask == cv2.GC_BGD]
    foreground = chromaticity[mask == cv2.GC_FGD]
    background_center = np.median(background, axis=0)
    foreground_center = np.median(foreground, axis=0)
    covariance = np.cov(background.T) + np.cov(foreground.T) + np.eye(2) * 0.001
    direction = np.linalg.solve(covariance, foreground_center - background_center)
    separation = (foreground_center - background_center) @ direction
    if separation <= 1.0:
        return
    score = ((chromaticity - background_center) @ direction) / separation
    luminance = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    background_luminance = np.median(luminance[mask == cv2.GC_BGD])
    shadow = (score < score_threshold) & (luminance < background_luminance * darkness_ratio)
    left, top, right, bottom = rectangle
    grid_y, grid_x = np.indices(mask.shape)
    # 어두운 알약 끝까지 배경으로 고정하지 않도록 긴 축의 양 끝은 보호한다.
    if bottom - top >= right - left:
        endpoint = (grid_y < top + (bottom - top) * 0.2) | (grid_y > bottom - (bottom - top) * 0.2)
    else:
        endpoint = (grid_x < left + (right - left) * 0.2) | (grid_x > right - (right - left) * 0.2)
    endpoint &= (grid_x >= left) & (grid_x < right) & (grid_y >= top) & (grid_y < bottom)
    mask[shadow & ~endpoint & (mask != cv2.GC_FGD)] = cv2.GC_BGD


def _segment_pill_grabcut(
    crop: np.ndarray,
    rectangle: tuple[int, int, int, int],
    seed_radius_ratio: float,
    iterations: int,
    seed: int,
) -> tuple[list[int], bool, int]:
    """중앙 알약의 밝기와 색상을 함께 분리하고 (몸체 박스, 탐색 경계 접촉 여부, 넓이)를 반환한다."""
    left, top, right, bottom = rectangle
    mask = np.full(crop.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    mask[top:bottom, left:right] = cv2.GC_PR_FGD
    mask[:3, :] = mask[-3:, :] = mask[:, :3] = mask[:, -3:] = cv2.GC_BGD
    center = (round((left + right) / 2), round((top + bottom) / 2))
    radii = (
        max(2, round((right - left) * seed_radius_ratio)),
        max(2, round((bottom - top) * seed_radius_ratio)),
    )
    cv2.ellipse(mask, center, radii, 0, 0, 360, cv2.GC_FGD, -1)
    _mark_shadow_background(crop, mask, rectangle)
    cv2.setRNGSeed(seed)
    cv2.grabCut(
        crop, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), iterations, cv2.GC_INIT_WITH_MASK
    )
    foreground = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    # 색이 다른 캡슐의 연결부가 얇게 끊어진 경우를 잇는다.
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, components, stats, _ = cv2.connectedComponentsWithStats(foreground)
    component_id = components[center[1], center[0]]
    if component_id == 0:
        raise ValueError("알약 중앙과 연결된 윤곽을 찾지 못했습니다.")
    left, top, width, height, area = map(int, stats[component_id])
    touches_edge = (
        left <= 3
        or top <= 3
        or left + width >= crop.shape[1] - 3
        or top + height >= crop.shape[0] - 3
    )
    return [left, top, width, height], touches_edge, area


def _measure_padding(
    img: np.ndarray,
    bbox: Sequence[float],
    padding_pixels: int = 4,
    iterations: int = 4,
    max_crop_size: int = 360,
    seed: int = 42,
) -> dict:
    """bbox 하나의 알약 몸체를 두 GrabCut 초기 조건으로 추정해 여백 비율을 잰다(JJAM 셀 54).

    두 추정의 합집합을 몸체(foreground_bbox)로 보고, 경계 여유를 더한 것을 tight_bbox로 둔다.
    """
    left, top, width, height = map(int, bbox)
    img_height, img_width = img.shape[:2]
    if min(width, height) <= 0 or min(left, top) < 0:
        return {"status": "invalid_bbox", "review_reason": "유효하지 않은 좌표"}
    if left + width > img_width or top + height > img_height:
        return {"status": "invalid_bbox", "review_reason": "이미지 범위 밖 원본 bbox"}
    margin = max(12, round(max(width, height) * 0.25))
    crop_left, crop_top = max(0, left - margin), max(0, top - margin)
    crop_right = min(img_width, left + width + margin)
    crop_bottom = min(img_height, top + height + margin)
    original_crop = img[crop_top:crop_bottom, crop_left:crop_right]
    scale = min(1.0, max_crop_size / max(original_crop.shape[:2]))
    crop = cv2.resize(original_crop, None, fx=scale, fy=scale)
    scale_x = crop.shape[1] / original_crop.shape[1]
    scale_y = crop.shape[0] / original_crop.shape[0]
    rectangle = (
        round((left - crop_left) * scale_x),
        round((top - crop_top) * scale_y),
        round((left + width - crop_left) * scale_x),
        round((top + height - crop_top) * scale_y),
    )
    try:
        first, first_edge, first_area = _segment_pill_grabcut(
            crop, rectangle, 0.12, iterations, seed
        )
        second, second_edge, second_area = _segment_pill_grabcut(
            crop, rectangle, 0.20, iterations, seed
        )
    except (cv2.error, ValueError) as error:
        return {"status": "segmentation_failed", "review_reason": str(error)}
    # 둘 중 더 작은 상자를 택하면 누락이 생길 수 있으므로 두 경계를 모두 포함한다.
    mask_left, mask_top = min(first[0], second[0]), min(first[1], second[1])
    mask_right = max(first[0] + first[2], second[0] + second[2])
    mask_bottom = max(first[1] + first[3], second[1] + second[3])
    # 경계 여유를 넣기 전의 몸체 사각형을 여백 비율의 기준으로 쓴다.
    body_left = max(0, int(np.floor(crop_left + mask_left / scale_x)))
    body_top = max(0, int(np.floor(crop_top + mask_top / scale_y)))
    body_right = min(img_width, int(np.ceil(crop_left + mask_right / scale_x)))
    body_bottom = min(img_height, int(np.ceil(crop_top + mask_bottom / scale_y)))
    foreground_bbox = [body_left, body_top, body_right - body_left, body_bottom - body_top]
    # 축소 좌표를 원본으로 복원할 때 바깥으로 반올림하고 추가 여유를 둔다.
    padding = max(padding_pixels, int(np.ceil(min(width, height) * 0.01)))
    tight_left, tight_top = max(0, body_left - padding), max(0, body_top - padding)
    tight_right = min(img_width, body_right + padding)
    tight_bottom = min(img_height, body_bottom + padding)
    tight = [tight_left, tight_top, tight_right - tight_left, tight_bottom - tight_top]
    agreement = bbox_iou(first, second)
    reasons = []
    if first_edge or second_edge:
        reasons.append("탐색 영역 경계 접촉: 알약 전체 포함 여부 검토")
    if agreement < 0.90:
        reasons.append("초기 조건에 따라 윤곽이 달라짐")
    foreground_ratio = max(first_area, second_area) / (width * height * scale_x * scale_y)
    if foreground_ratio < 0.15 or foreground_ratio > 1.1:
        reasons.append("전경 크기 이상: 배경 혼입 또는 알약 누락 검토")
    if tight[2] * tight[3] < width * height * 0.5:
        reasons.append("원본 대비 과도한 축소: 실제 여백 또는 윤곽 누락 검토")
    status = "review_required" if reasons else "estimated"
    if status == "estimated" and (
        tight_left < left - 3
        or tight_top < top - 3
        or tight_right > left + width + 3
        or tight_bottom > top + height + 3
    ):
        status = "extends_outside"
        reasons.append("추정 경계가 원본 bbox 밖까지 이어짐: 잘림/그림자 검토")
    return {
        "status": status,
        "tight_bbox": tight,
        "foreground_bbox": foreground_bbox,
        "padding_ratio": max(0.0, 1 - foreground_bbox[2] * foreground_bbox[3] / (width * height)),
        "segmentation_agreement": agreement,
        "review_reason": "; ".join(reasons),
    }


def measure_bbox_padding(ann_df: pd.DataFrame, img_dir: Path, seed: int = 42) -> pd.DataFrame:
    """bbox마다 GrabCut으로 알약 몸체를 추정해 여백 비율을 잰다(JJAM 셀 54). 수 분 걸린다.

    한 행 = bbox. status가 estimated인 행만 여백 집계에 쓴다. 나머지(review_required,
    extends_outside, invalid_bbox, segmentation_failed, missing_image)는 경계 확인 대상이다.
    padding_ratio = 1 − 몸체 넓이 ÷ bbox 넓이, tight_bbox = 몸체 + 경계 여유(새 bbox 후보),
    bbox_iou = 원본 bbox와 tight_bbox의 IoU. 결과가 같도록 OpenCV를 단일 스레드로 돌린다.
    """
    keys = ["source_json", "annotation_id", "file_name", "category_id"]
    rows = []
    n_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        for file_name, group in ann_df.dropna(subset=BBOX_COLS).groupby("file_name"):
            img = cv2.imread(str(img_dir / file_name))
            for label in group[[*keys, *BBOX_COLS]].itertuples(index=False):
                bbox = [label.x, label.y, label.w, label.h]
                if img is None:
                    result = {"status": "missing_image", "review_reason": "사진 없음"}
                else:
                    result = _measure_padding(img, bbox, seed=seed)
                tight = result.get("tight_bbox")
                row = {k: getattr(label, k) for k in keys}
                row["bbox"] = [int(v) for v in bbox]
                row.update(result)
                row["bbox_iou"] = bbox_iou(bbox, tight) if tight else float("nan")
                rows.append(row)
    finally:
        cv2.setNumThreads(n_threads)
    cols = [*keys, "bbox", "status", "tight_bbox", "foreground_bbox", "padding_ratio"]
    cols += ["segmentation_agreement", "bbox_iou", "review_reason"]
    return pd.DataFrame(rows).reindex(columns=cols)
