"""02_eda에서 확정한 이미지 선택과 bbox 좌표를 학습 입력에 연결한다."""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data_process.data_processing import bbox_flags, find_box_pairs, load_annotations


def _selected_sources(
    selected: pd.DataFrame, folders: list[Path]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """최종 CSV의 경로로 이미지를 찾고 대상 및 동반 알약 JSON만 읽는다."""
    folder_map = {folder.name: folder for folder in folders}
    if len(folder_map) != len(folders):
        raise ValueError("외부 데이터 폴더 이름이 중복됩니다.")
    if "path" not in selected:
        raise ValueError("최종 CSV에 이미지 path 열이 필요합니다.")
    image_rows = selected[["file_name", "path"]].drop_duplicates()
    if image_rows["file_name"].duplicated().any():
        raise ValueError("같은 이미지에 서로 다른 경로가 지정되어 있습니다.")
    images = []
    for row in tqdm(image_rows.itertuples(), total=len(image_rows), desc="확정 이미지 경로 확인"):
        parts = Path(str(row.path).replace("\\", "/")).parts
        matches = [i for i, part in enumerate(parts) if part in folder_map]
        if len(matches) != 1:
            raise ValueError(f"현재 외부 데이터 폴더에 연결할 수 없는 경로: {row.path}")
        index = matches[0]
        path = folder_map[parts[index]].joinpath(*parts[index + 1 :])
        if path.name != row.file_name or not path.is_file():
            raise FileNotFoundError(f"최종 이미지가 없습니다: {path}")
        images.append({"file_name": row.file_name, "path": path})

    # 파일명에 명시된 모든 약의 JSON을 포함해 동반 라벨도 누락시키지 않는다.
    targets: dict[str, set[Path]] = {}
    for row in selected[["file_name", "source_json"]].itertuples():
        relative = Path(str(row.source_json).replace("\\", "/"))
        if ".." in relative.parts or relative.is_absolute():
            raise ValueError(f"잘못된 원본 JSON 상대 경로: {relative}")
        folder_name = relative.parts[0]
        if folder_name not in folder_map:
            raise FileNotFoundError(f"원본 라벨 폴더가 없습니다: {folder_name}")
        original = folder_map[folder_name].joinpath(*relative.parts[1:])
        if original.stem != Path(row.file_name).stem:
            raise ValueError("이미지 이름과 원본 JSON 이름이 다릅니다.")
        targets.setdefault(folder_name, set()).add(original)
        for code in row.file_name.split("_")[0].split("-")[1:]:
            targets[folder_name].add(original.parent.parent / f"K-{code}" / original.name)

    annotations, errors = [], []
    for name, paths in tqdm(targets.items(), desc="선택 이미지 JSON 읽기"):
        table, failures = load_annotations(folder_map[name], json_paths=list(paths))
        if not table.empty:
            table["source_json"] = name + "/" + table["source_json"]
            table["category_id_raw"] = table["category_id"]
            table["category_id"] = pd.to_numeric(
                table["drug_N"].astype(str).str.removeprefix("K-"), errors="coerce"
            ).astype("Int64")
            table["category_name"] = table["dl_name"]
            table["zip_name"] = name
            annotations.append(table)
        if not failures.empty:
            failures["source_json"] = name + "/" + failures["source_json"]
            errors.append(failures)
    return (
        pd.DataFrame(images),
        pd.concat(annotations, ignore_index=True) if annotations else pd.DataFrame(),
        pd.concat(errors, ignore_index=True) if errors else pd.DataFrame(),
    )


def load_reviewed_aihub(
    selection_path: Path, folders: list[Path], competition_labels: pd.DataFrame
) -> dict:
    """최종 선택을 재추출하지 않고 동반 라벨에 검수 완료 좌표를 반영한다."""
    if not selection_path.is_file():
        raise FileNotFoundError(f"02_eda 최종 CSV가 없습니다: {selection_path}")
    selected = pd.read_csv(selection_path)
    identity = ["file_name", "source_json", "annotation_id", "category_id"]
    numeric_columns = ["x", "y", "w", "h", "width", "height"]
    required = set(identity + numeric_columns + ["dl_name"])
    if required - set(selected.columns):
        raise ValueError(f"최종 CSV 필수 열 누락: {sorted(required - set(selected.columns))}")
    if selected.empty or selected.duplicated(identity).any():
        raise ValueError("최종 선택이 비어 있거나 bbox 식별자가 중복됩니다.")
    competition_ids = set(competition_labels["category_id"].astype(int))
    if not set(selected["category_id"]).issubset(competition_ids):
        raise ValueError("최종 선택 CSV에 대회 클래스가 아닌 약 코드가 있습니다.")

    image_index, annotations, errors = _selected_sources(selected, folders)
    if image_index["file_name"].duplicated().any():
        raise ValueError("외부 이미지 파일명이 중복됩니다.")
    paths = image_index.set_index("file_name")["path"]
    files = sorted(selected["file_name"].unique())
    missing = set(files) - set(paths.index)
    if missing:
        raise FileNotFoundError(f"최종 이미지 {len(missing)}장 누락: {sorted(missing)[:5]}")

    # 최종 선택과 동반 알약의 JSON만 읽었으므로 로드 실패는 모두 관련 오류다.
    failed = errors[errors["excluded"]].copy() if not errors.empty else errors.copy()
    relevant = failed.copy()
    if not relevant.empty:
        details = relevant[["source_json", "stage", "message"]].head(5).to_dict("records")
        raise ValueError(f"선택 이미지의 원본 JSON 로드 실패 {len(relevant)}건: {details}")
    if annotations.empty:
        raise ValueError("읽을 수 있는 외부 annotation이 없습니다.")
    boxes = annotations[annotations["file_name"].isin(files)].copy().reset_index(drop=True)
    if boxes.duplicated(identity).any():
        raise ValueError("원본 annotation의 식별자가 중복됩니다.")
    indices = pd.MultiIndex.from_frame(boxes[identity]).get_indexer(
        pd.MultiIndex.from_frame(selected[identity])
    )
    if (indices < 0).any():
        raise ValueError("최종 선택 bbox와 원본 JSON 식별자가 일치하지 않습니다.")
    if not np.array_equal(
        boxes.loc[indices, ["width", "height"]].to_numpy(),
        selected[["width", "height"]].to_numpy(),
    ):
        raise ValueError("최종 CSV와 원본 JSON의 이미지 크기가 다릅니다.")

    # 원본에서 동반 알약을 유지하고 선택 bbox에만 검수 완료 좌표를 적용한다.
    boxes[["x", "y", "w", "h"]] = boxes[["x", "y", "w", "h"]].astype(float)
    boxes.loc[indices, ["x", "y", "w", "h"]] = selected[["x", "y", "w", "h"]].to_numpy()
    boxes["area"] = boxes["w"] * boxes["h"]
    selected["zip_name"] = boxes.loc[indices, "zip_name"].to_numpy()
    selected["path"] = selected["file_name"].map(paths)
    selected["area"] = selected["w"] * selected["h"]
    if (
        not np.isfinite(boxes[numeric_columns].to_numpy(dtype=float)).all()
        or (boxes[["w", "h", "width", "height"]] <= 0).any().any()
        or bbox_flags(boxes)["out_of_image"].any()
        or boxes.duplicated(["file_name", "category_id"]).any()
    ):
        raise ValueError("최종 결과 또는 동반 라벨에 좌표·크기·중복 오류가 있습니다.")
    labeled = boxes.groupby("file_name")["category_id"].agg(set)
    for name in files:
        codes = {int(code) for code in name.split("_")[0].split("-")[1:]}
        if codes != labeled.get(name, set()):
            raise ValueError(f"파일명과 전체 이미지 라벨 클래스가 다릅니다: {name}")
    pairs = find_box_pairs(boxes, max_gap=0)
    # 확정 이미지는 다시 제외하거나 새 이미지로 보충하지 않는다.
    reasons = pd.DataFrame({"추가 제외": False}, index=pd.Index(files, name="file_name"))
    return {
        "n_images": len(image_index),
        "n_labels": len(annotations),
        "common_classes": sorted(selected["category_id"].unique()),
        "selected": selected,
        "labels": boxes[boxes["category_id"].isin(competition_ids)].copy(),
        "background": boxes[~boxes["category_id"].isin(competition_ids)].copy(),
        "reasons": reasons,
        "paths": paths,
        "remaining_overlaps": pairs[pairs["iou"].gt(0)].copy(),
        "unrelated_load_errors": failed.iloc[:0].copy(),
    }
