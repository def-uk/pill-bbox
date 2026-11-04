"""알약 메타데이터에서 사람이 확인할 OCR 각인 후보 표를 만든다."""

import hashlib
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch import Tensor

from src.config import PROJECT_ROOT


def true_values(values: pd.Series) -> pd.Series:
    """CSV에서 읽은 문자열 False를 승인으로 해석하지 않도록 불리언을 변환한다."""
    return values.fillna(False).astype(str).str.lower().isin(["true", "1"])


def read_ocr_review_table(path: Path) -> pd.DataFrame:
    """숫자로만 된 각인의 선행 0을 보존하며 OCR 검수표를 읽는다."""
    return pd.read_csv(
        path,
        keep_default_na=False,
        dtype={
            column: str
            for column in [
                "file_name",
                "source_json",
                "text",
                "decision",
                "suggested_text",
                "print_front",
                "print_back",
            ]
        },
    )


def attach_competition_metadata(labels: pd.DataFrame, annotation_dir: Path) -> pd.DataFrame:
    """대회 보정 bbox는 유지하고 원본 JSON에서 일치하는 각인 메타데이터만 연결한다."""
    from data_process.data_processing import load_annotations

    original, errors = load_annotations(annotation_dir)
    if not errors.empty and errors["excluded"].any():
        raise ValueError("대회 원본 JSON 로드 실패로 각인 메타데이터를 연결할 수 없습니다.")
    identity = ["file_name", "source_json", "annotation_id", "category_id"]
    fields = ["drug_dir", "print_front", "print_back"]
    if set(identity + fields) - set(original.columns):
        raise ValueError("대회 원본 JSON에 각인 메타데이터가 없습니다.")
    original = _normalize_identity(original)
    result = (
        _normalize_identity(labels)
        .drop(columns=fields, errors="ignore")
        .merge(
            original[identity + fields],
            on=identity,
            how="left",
            validate="one_to_one",
            indicator="_ocr_metadata_match",
        )
    )
    # 추가 보정으로 생긴 bbox의 방향·문자는 다른 알약에서 추정해 채우지 않는다.
    result["metadata_origin"] = (
        result["_ocr_metadata_match"]
        .map({"both": "competition_json", "left_only": "unavailable"})
        .astype(str)
    )
    return result.drop(columns="_ocr_metadata_match")


def _normalize_identity(rows: pd.DataFrame) -> pd.DataFrame:
    """CSV 왕복 시 빈 원본 경로와 숫자 식별자의 표현을 통일한다."""
    rows = rows.copy()
    rows["source_json"] = (
        rows["source_json"].fillna("").astype(str).str.replace("\\", "/", regex=False)
    )
    if rows["file_name"].isna().any() or rows["file_name"].eq("").any():
        raise ValueError("OCR 검수표에 이미지 파일명이 없는 행이 있습니다.")
    rows["file_name"] = rows["file_name"].astype(str)
    for column in ["annotation_id", "category_id"]:
        rows[column] = pd.to_numeric(rows[column], errors="raise").astype("Int64")
        if rows[column].isna().any():
            raise ValueError(f"OCR 검수표의 {column} 식별자가 비어 있습니다.")
    return rows


def _deduplicate_reviews(rows: pd.DataFrame) -> pd.DataFrame:
    """동일 bbox 중복에서 사람 판정을 보존하고 좌표·판정 충돌은 차단한다."""
    identity = ["file_name", "source_json", "annotation_id", "category_id"]
    duplicated = rows.duplicated(identity, keep=False)
    if not duplicated.any():
        return rows
    merged = []
    for key, group in rows[duplicated].groupby(identity, dropna=False, sort=False):
        coordinates = group[["x", "y", "w", "h"]].to_numpy(dtype=float)
        if not np.allclose(coordinates, coordinates[0], atol=1e-6, rtol=0):
            raise ValueError(f"중복 OCR 항목의 bbox 좌표가 충돌합니다: {key}")
        has_work = group["text"].ne("") | group["decision"].ne("") | true_values(group["reviewed"])
        choices = group[has_work].copy() if has_work.any() else group.copy()
        if len(choices[["text", "decision"]].drop_duplicates()) > 1:
            raise ValueError(f"중복 OCR 항목의 사람 판정이 충돌합니다: {key}")
        choices["_review_priority"] = true_values(choices["reviewed"])
        choices["_active_priority"] = (
            true_values(choices["active"]) if "active" in choices else True
        )
        chosen = choices.sort_values(["_review_priority", "_active_priority"]).iloc[-1]
        merged.append(chosen.drop(labels=["_review_priority", "_active_priority"]))
    return pd.concat([rows[~duplicated], pd.DataFrame(merged)], ignore_index=True)


def build_ocr_review_table(annotations: pd.DataFrame) -> pd.DataFrame:
    """사진 방향에 맞는 각인을 후보로만 기록하고 승인·정답은 빈 값으로 남긴다."""
    identity = ["file_name", "source_json", "annotation_id", "category_id"]
    required = identity + ["x", "y", "w", "h", "drug_dir", "print_front", "print_back"]
    if set(required) - set(annotations.columns):
        raise ValueError(
            f"각인 검수표 필수 열 누락: {sorted(set(required) - set(annotations.columns))}"
        )
    annotations = _normalize_identity(annotations)
    if annotations.duplicated(identity).any():
        raise ValueError("각인 후보의 bbox 식별자가 중복됩니다.")
    table = annotations[required].copy().reset_index(drop=True)
    for column in ["drug_dir", "print_front", "print_back"]:
        table[column] = table[column].fillna("").astype(str).str.strip()
    table["suggested_text"] = ""
    table.loc[table["drug_dir"].eq("앞면"), "suggested_text"] = table["print_front"]
    table.loc[table["drug_dir"].eq("뒷면"), "suggested_text"] = table["print_back"]
    table["candidate_status"] = "사진에서 각인 확인 필요"
    table.loc[table["suggested_text"].eq(""), "candidate_status"] = "문자 후보 없음·방향 확인 필요"
    symbols = table["suggested_text"].str.contains("마크|분할선", regex=True)
    table.loc[symbols, "candidate_status"] = "문자 외 마크·분할선 포함: 직접 확인 필요"
    table["review_id"] = [
        hashlib.sha256("\x1f".join(map(str, values)).encode()).hexdigest()[:24]
        for values in table[identity].itertuples(index=False, name=None)
    ]
    # 메타데이터가 실제 사진의 OCR 정답이라고 자동 확정하지 않는다.
    table["text"] = ""
    table["decision"] = ""
    table["reviewed"] = False
    table["active"] = True
    if "path" in annotations:
        table["path"] = annotations["path"].astype(str).to_numpy()
    if "metadata_origin" in annotations:
        table["metadata_origin"] = annotations["metadata_origin"].to_numpy()
    return table


def save_ocr_review_table(candidates: pd.DataFrame, path: Path) -> pd.DataFrame:
    """기존 사람 판정은 보존하고 새 후보를 추가해 검수표를 저장한다."""
    identity = ["file_name", "source_json", "annotation_id", "category_id"]
    table = _normalize_identity(candidates)
    if table.duplicated(identity).any():
        raise ValueError("새 OCR 후보의 bbox 식별자가 중복됩니다.")
    backup_needed = False
    if path.is_file():
        previous = _normalize_identity(read_ocr_review_table(path))
        backup_needed = previous.duplicated(identity).any()
        previous = _deduplicate_reviews(previous)
        old = previous.set_index(identity)
        current = table.set_index(identity)
        shared = current.index.intersection(old.index)
        for column in ["x", "y", "w", "h"]:
            if not np.allclose(
                current.loc[shared, column].to_numpy(dtype=float),
                old.loc[shared, column].to_numpy(dtype=float),
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError("기존 검수 bbox 좌표가 달라졌습니다. 검수표를 별도로 보관하세요.")
        for column in ["text", "decision", "reviewed"]:
            current[column] = current[column].astype(object)
            current.loc[shared, column] = old.loc[shared, column]
        inactive = old.loc[~old.index.isin(current.index)].copy()
        inactive["active"] = False
        table = pd.concat([current, inactive]).reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    if table.duplicated(identity).any():
        raise ValueError("OCR 검수표 병합 후 식별자 중복이 발생했습니다. 저장하지 않습니다.")
    if backup_needed:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = path.with_name(f"{path.stem}_before_dedup_{stamp}.csv")
        backup.write_bytes(path.read_bytes())
    temporary = path.with_suffix(".tmp")
    table.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)
    return table


def external_ocr_split(files: set[str], val_fraction: float, seed: int) -> pd.DataFrame:
    """같은 조합을 묶고 목표 이미지 비율에 가깝게 외부 OCR train/val을 나눈다."""
    if not 0 < val_fraction < 1:
        raise ValueError("외부 OCR 검증 비율은 0과 1 사이여야 합니다.")
    table = pd.DataFrame({"file_name": sorted(files)})
    table["combo"] = table["file_name"].str.split("_").str[0]
    counts = table.groupby("combo").size()
    if len(counts) < 2:
        raise ValueError("외부 OCR 조합이 최소 2개 있어야 train/val 분리가 가능합니다.")
    target = max(1, min(len(table) - 1, round(len(table) * val_fraction)))
    selected, current = set(), 0
    for combo in np.random.default_rng(seed).permutation(counts.index.to_numpy()):
        size = int(counts[combo])
        if abs(current + size - target) < abs(current - target):
            selected.add(combo)
            current += size
    if not selected:
        selected.add((counts - target).abs().idxmin())
    if len(selected) == len(counts):
        raise ValueError("조합을 유지하면서 외부 train/val을 나눌 수 없습니다.")
    table["split"] = np.where(table["combo"].isin(selected), "val", "train")
    return table


def reviewed_ocr_splits(
    review: pd.DataFrame,
    split_path: Path,
    external_files: set[str],
    *,
    label_mode: str = "reviewed",
    external_val_fraction: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """명시한 정답 출처로 OCR 문자열을 만들고 YOLO train/val에 배정한다."""
    if label_mode not in {"reviewed", "metadata"}:
        raise ValueError("OCR label_mode는 reviewed 또는 metadata여야 합니다.")
    if not 0 <= external_val_fraction < 1:
        raise ValueError("외부 OCR 검증 비율은 0 이상 1 미만이어야 합니다.")
    required = {"reviewed", "decision", "text", "file_name", "path"}
    if required - set(review.columns):
        raise ValueError(f"OCR 검수표 필수 열 누락: {sorted(required - set(review.columns))}")
    active = true_values(review["active"]) if "active" in review else True
    reviewed = true_values(review["reviewed"]) & review["decision"].eq("승인") & active
    approved = review[reviewed].copy()
    approved["label_source"] = "human_reviewed"
    if label_mode == "metadata":
        if "suggested_text" not in review:
            raise ValueError("JSON 각인 후보 suggested_text가 없습니다.")
        suggested = review["suggested_text"].fillna("").astype(str).str.strip()
        weak = (
            ~true_values(review["reviewed"])
            & review["decision"].eq("")
            & active
            & suggested.ne("")
            & ~suggested.str.contains("마크|분할선|로고|없음|무각인|^[\\W_]+$", regex=True)
        )
        metadata = review[weak].copy()
        metadata["text"] = suggested[weak]
        metadata["label_source"] = "json_metadata"
        approved = pd.concat([approved, metadata], ignore_index=True)
    approved["text"] = approved["text"].fillna("").astype(str).str.strip()
    approved = approved[approved["text"].ne("")].copy()
    split = pd.read_csv(split_path)
    if split["file_name"].duplicated().any():
        raise ValueError("YOLO 이미지 분할에 중복이 있습니다.")
    mapping = split.set_index("file_name")["split"].to_dict()
    if set(mapping) & external_files:
        raise ValueError("대회와 외부 이미지 이름이 겹칩니다.")
    approved["split"] = approved["file_name"].map(mapping)
    is_external = approved["file_name"].isin(external_files)
    approved["data_source"] = np.where(is_external, "external", "competition")
    approved.loc[is_external, "split"] = "train"
    if external_val_fraction and is_external.any():
        external = external_ocr_split(
            set(approved.loc[is_external, "file_name"]), external_val_fraction, seed
        ).set_index("file_name")["split"]
        approved.loc[is_external, "split"] = approved.loc[is_external, "file_name"].map(external)
    if approved["split"].isna().any():
        raise ValueError("분할에 없는 OCR 이미지가 있습니다.")
    if not set(approved["split"]).issubset({"train", "val"}):
        raise ValueError("OCR은 지정한 fold의 train/val 분할만 사용합니다.")
    approved["combo"] = approved["file_name"].str.split("_").str[0]
    # 검수된 행뿐 아니라 검증 이미지 전체의 조합과 비교한다.
    val_combos = set(split.loc[split["split"].eq("val"), "file_name"].str.split("_").str[0])
    train_combos = set(approved.loc[approved["split"].eq("train"), "combo"])
    if train_combos & val_combos:
        raise ValueError("OCR train과 YOLO val의 알약 조합이 겹칩니다.")
    if train_combos & set(approved.loc[approved["split"].eq("val"), "combo"]):
        raise ValueError("OCR train과 val에 같은 알약 조합이 있습니다.")
    approved = approved.reset_index(drop=True)
    approved.attrs["external_val_fraction"] = external_val_fraction
    approved.attrs["split_seed"] = seed
    return approved


def pill_crop(image: Image.Image, bbox: list[float]) -> Image.Image:
    """원본 픽셀 xywh 좌표를 검증하고 알약 이미지를 자른다."""
    if len(bbox) != 4 or not np.isfinite(bbox).all():
        raise ValueError("잘못된 OCR crop 좌표입니다.")
    x, y, width, height = bbox
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("OCR bbox의 위치 또는 크기가 잘못됐습니다.")
    if x + width > image.width or y + height > image.height:
        raise ValueError("OCR bbox가 이미지 밖으로 나갑니다.")
    return image.crop((math.floor(x), math.floor(y), math.ceil(x + width), math.ceil(y + height)))


def ocr_pixels(image: Image.Image, height: int, width: int) -> np.ndarray:
    """종횡비와 오른쪽 여백을 유지한 uint8 OCR 입력을 만든다."""
    if height < 4 or width < 4:
        raise ValueError("OCR 입력 크기는 4픽셀 이상이어야 합니다.")
    image = image.convert("L")
    target_width = min(width, max(1, round(image.width * height / image.height)))
    image = image.resize((target_width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (width, height), color=255)
    canvas.paste(image, (0, 0))
    return np.asarray(canvas, dtype=np.uint8).copy()


def ocr_tensor(image: Image.Image, height: int, width: int) -> Tensor:
    """OCR 입력 픽셀을 -1~1 범위의 텐서로 정규화한다."""
    import torch

    return torch.from_numpy(ocr_pixels(image, height, width).astype(np.float32))[None] / 127.5 - 1.0


def prepare_crop_cache(rows: pd.DataFrame, path: Path, height: int, width: int) -> None:
    """원본 이미지는 한 번씩만 열고 작은 OCR 입력을 디스크 메모리맵에 저장한다."""
    from tqdm.auto import tqdm

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    cache = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.uint8, shape=(len(rows), height, width)
    )
    try:
        groups = rows.groupby("path", sort=False)
        for source, group in tqdm(groups, total=len(groups), desc="OCR crop 준비", mininterval=5):
            with Image.open(image_path(str(source))) as image:
                for _, row in group.iterrows():
                    crop = pill_crop(image, row[["x", "y", "w", "h"]].to_list())
                    cache[int(row["_crop_index"])] = ocr_pixels(crop, height, width)
        cache.flush()
    except BaseException:
        del cache
        temporary.unlink(missing_ok=True)
        raise
    del cache
    temporary.replace(path)


def image_path(value: str) -> Path:
    """프로젝트 상대 경로 또는 실행 환경에서 생성한 이미지 경로를 읽는다."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def review_ocr_interactively(path: Path, limit: int = 20, seed: int = 42) -> None:
    """알약 crop을 보며 후보 승인·문자 직접 입력·제외 결정을 한 건씩 저장한다."""
    import matplotlib.pyplot as plt
    from IPython.display import clear_output

    table = read_ocr_review_table(path)
    done = true_values(table["reviewed"]) & table["decision"].isin(["승인", "거부"])
    active = true_values(table["active"]) if "active" in table else True
    pending = table[~done & active]
    pending = pending.sample(n=min(limit, len(pending)), random_state=seed)
    for index, row in pending.iterrows():
        with Image.open(image_path(str(row["path"]))) as image:
            crop = pill_crop(image, row[["x", "y", "w", "h"]].to_list())
        clear_output(wait=True)
        plt.figure(figsize=(9, 5))
        plt.imshow(crop)
        plt.axis("off")
        plt.show()
        plt.close()
        print(f"클래스 {row['category_id']} / {row['drug_dir']} / {row['file_name']}")
        print(f"메타데이터 후보: {row['suggested_text']!r}")
        print("y: 후보 승인 | t:문자열: 직접 입력 | n: 제외 | s: 보류 | q: 종료")
        while True:
            answer = input("OCR 판정: ").strip()
            if answer in {"q", "s", "n"}:
                break
            text = str(row["suggested_text"]).strip() if answer == "y" else ""
            if answer.startswith("t:"):
                text = answer[2:].strip()
            if answer == "y" and any(word in text for word in ["마크", "분할선"]):
                print("마크·분할선 설명은 문자 정답이 아닙니다. t:로 실제 문자를 입력하세요.")
                continue
            if text:
                break
            print("보이는 각인을 t:문자열 형식으로 입력하거나 n/s/q를 선택하세요.")
        if answer == "q":
            break
        if answer == "s":
            continue
        table.at[index, "text"] = "" if answer == "n" else text
        table.at[index, "decision"] = "거부" if answer == "n" else "승인"
        table.at[index, "reviewed"] = True
        temporary = path.with_suffix(".tmp")
        table.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(path)
    print(f"OCR 검수 기록: {path}")
