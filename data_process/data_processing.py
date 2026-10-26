import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg"}

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
