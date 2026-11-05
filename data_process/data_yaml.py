from pathlib import Path

import pandas as pd
import yaml

from src.config import PROJECT_ROOT

YOLO_ROOT = PROJECT_ROOT / "data" / "processed" / "yolo"
CLASS_MAP_COLS = ["yolo_idx", "category_id", "dl_name"]


def yolo_dir(label_version: str) -> Path:
    """라벨 버전의 YOLO 데이터 폴더(images/, labels/, class_map.csv, splits/)."""
    return YOLO_ROOT / label_version


def data_yaml_path(label_version: str, split_name: str, fold: int) -> Path:
    """학습에 넘길 fold별 data.yaml 경로."""
    return yolo_dir(label_version) / "splits" / split_name / f"fold_{fold}" / "data.yaml"


def write_full_train_files(source_yaml: Path, out_dir: Path) -> Path:
    """기존 train·val 목록을 합쳐 검증 없는 전체 학습용 입력을 별도로 만든다."""
    source_yaml = source_yaml.resolve()
    data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", source_yaml.parent))
    if not root.is_absolute():
        root = source_yaml.parent / root
    images = set()
    counts = {}
    for split in ["train", "val"]:
        lists = data[split] if isinstance(data[split], list) else [data[split]]
        selected = set()
        for value in lists:
            path = Path(value)
            path = path if path.is_absolute() else root / path
            if path.suffix != ".txt":
                raise ValueError("전체 학습 변환은 기존 파이프라인 txt 목록만 지원합니다.")
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                image = Path(line.strip())
                image = image if image.is_absolute() else path.parent / image
                if not image.is_file():
                    raise FileNotFoundError(image)
                selected.add(str(image.absolute()))
        counts[split] = len(selected)
        images.update(selected)
    if not images or out_dir.resolve() == source_yaml.parent:
        raise ValueError("빈 입력 또는 원본 분할 덮어쓰기는 허용하지 않습니다.")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = (out_dir / "train.txt").resolve()
    train_path.write_text("".join(f"{path}\n" for path in sorted(images)), encoding="utf-8")
    # Ultralytics 스키마용 경로이며 FullTrainTrainer는 검증을 호출하지 않는다.
    data.update(path=str(root.resolve()), train=str(train_path), val=str(train_path))
    output = out_dir / "data.yaml"
    output.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    import json

    (out_dir / "full_train_manifest.json").write_text(
        json.dumps(
            {
                "source_yaml": str(source_yaml),
                "source_counts": counts,
                "train_images": len(images),
                "validation": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def write_holdout_files(
    source_yaml: Path,
    out_dir: Path,
    val_fraction: float,
    seed: int,
    fixed_train_split: Path | None = None,
    external_train_only: bool = False,
) -> Path:
    """기존 입력의 대회·외부 이미지를 조합 단위로 나눠 공유 검증 분할을 저장한다."""
    import json

    import numpy as np

    if not 0 < val_fraction < 1:
        raise ValueError("검증 비율은 0과 1 사이여야 합니다.")
    # 전체 목록 생성도 파이프라인 안에서 수행하고 원본 fold 파일은 보존한다.
    combined = write_full_train_files(source_yaml, out_dir / "input_union")
    data = yaml.safe_load(combined.read_text(encoding="utf-8"))
    paths = Path(data["train"]).read_text(encoding="utf-8").splitlines()
    table = pd.DataFrame({"path": paths})
    table["file_name"] = table["path"].map(lambda value: Path(value).name)
    if table["file_name"].duplicated().any():
        raise ValueError("분할 입력에 같은 파일명의 서로 다른 이미지가 있습니다.")
    table["combo"] = table["file_name"].str.split("_").str[0]
    comp_dir = Path(data["path"]) / "images/train"
    table["data_source"] = table["path"].map(
        lambda value: "competition" if Path(value).parent == comp_dir else "external"
    )
    table["split"] = "train"
    fixed_combos = set()
    if fixed_train_split is not None:
        original = pd.read_csv(fixed_train_split)
        fixed = original["fixed"].astype(str).str.lower().isin(["true", "1"])
        fixed_combos = set(original.loc[fixed, "file_name"].str.split("_").str[0])
    table["fixed"] = table["combo"].isin(fixed_combos)
    for source, rows in table.groupby("data_source"):
        if external_train_only and source == "external":
            continue
        counts = rows.loc[~rows["fixed"]].groupby("combo").size()
        if len(counts) < 2:
            raise ValueError(f"{source}: 조합이 2개 이상이어야 검증 분할이 가능합니다.")
        target = max(1, min(len(rows) - 1, round(len(rows) * val_fraction)))
        selected, current = set(), 0
        for combo in np.random.default_rng(seed).permutation(counts.index.to_numpy()):
            size = int(counts[combo])
            if abs(current + size - target) < abs(current - target):
                selected.add(combo)
                current += size
        if not selected:
            selected.add((counts - target).abs().idxmin())
        if len(selected) == len(counts):
            raise ValueError(f"{source}: 조합을 유지한 train/val 분할이 불가능합니다.")
        table.loc[rows.index[rows["combo"].isin(selected)], "split"] = "val"
    train_combos = set(table.loc[table["split"].eq("train"), "combo"])
    val_combos = set(table.loc[table["split"].eq("val"), "combo"])
    if train_combos & val_combos:
        raise ValueError("대회·외부 데이터 사이의 조합 중복으로 분할 누수가 발생합니다.")
    table.to_csv(out_dir / "image_split.csv", index=False)
    for split in ["train", "val"]:
        path = (out_dir / f"{split}.txt").resolve()
        path.write_text(
            "".join(f"{value}\n" for value in table.loc[table["split"].eq(split), "path"]),
            encoding="utf-8",
        )
        data[split] = str(path)
    output = out_dir / "data.yaml"
    output.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    summary = {
        "seed": seed,
        "requested_val_fraction": val_fraction,
        "external_train_only": external_train_only,
        "competition_val_fraction": float(
            table.loc[table["data_source"].eq("competition"), "split"].eq("val").mean()
        ),
        "actual_val_fraction": float(table["split"].eq("val").mean()),
        "counts": table.groupby(["data_source", "split"])
        .size()
        .rename("images")
        .reset_index()
        .to_dict("records"),
        "combo_overlap": 0,
        "fixed_train_images": int(table["fixed"].sum()),
        "source_yaml": str(source_yaml.resolve()),
    }
    (out_dir / "holdout_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def build_class_map(labels: pd.DataFrame) -> pd.DataFrame:
    """category_id 오름차순으로 0부터 yolo_idx를 붙인 클래스 표를 만든다."""
    first = labels.sort_values("category_id").drop_duplicates("category_id")
    class_map = first[["category_id", "dl_name"]].reset_index(drop=True)
    class_map.insert(0, "yolo_idx", range(len(class_map)))
    return class_map[CLASS_MAP_COLS]


def load_class_map(label_version: str) -> pd.DataFrame:
    """저장된 class_map.csv를 읽는다. 예측의 yolo_idx를 category_id로 되돌릴 때 쓴다."""
    return pd.read_csv(yolo_dir(label_version) / "class_map.csv", encoding="utf-8")


def class_names(class_map: pd.DataFrame, use_code: bool = False) -> dict[int, str]:
    """data.yaml의 names. 기본은 한글 약 이름, use_code면 category_id 문자열이다.

    한글 폰트가 없는 환경(#50)에서 혼동 행렬 글자가 깨지면 use_code=True로 만든다.
    """
    col = "category_id" if use_code else "dl_name"
    return {
        int(i): str(name) for i, name in zip(class_map["yolo_idx"], class_map[col], strict=True)
    }


def write_fold_files(
    fold_rows: pd.DataFrame,
    images_dir: Path,
    out_dir: Path,
    names: dict[int, str],
    header: dict[str, object],
) -> Path:
    """fold 표(fold_k.csv 내용)로 train.txt·val.txt·data.yaml을 쓰고 data.yaml 경로를 반환한다.

    txt에는 images_dir 아래 이미지의 절대 경로를 적는다. ultralytics는 경로의 images/를
    labels/로 바꿔 라벨을 찾는다. header는 data.yaml 맨 위에 주석으로 남긴다(라벨 버전·분할·fold).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"path": str(images_dir.parent.parent)}
    for split in ["train", "val"]:
        files = sorted(fold_rows.loc[fold_rows["split"] == split, "file_name"])
        txt_path = out_dir / f"{split}.txt"
        txt_path.write_text("".join(f"{images_dir / f}\n" for f in files), encoding="utf-8")
        data[split] = str(txt_path)
    data.update(nc=len(names), names=names)
    comments = "".join(f"# {key}: {value}\n" for key, value in header.items())
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        comments + yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return yaml_path
