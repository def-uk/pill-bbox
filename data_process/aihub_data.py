"""AI Hub 경구약제 이미지 데이터(dataSetSn=576)의 조합 zip을 찾고 풀고 읽는다.

Colab에서 torch·ultralytics 없이도 import되도록 src.config를 import하지 않는다.
경로는 노트북이 정해서 넘긴다.
"""

import re
import shutil
import unicodedata
import zipfile
from collections.abc import Iterable
from pathlib import Path

import gdown
import pandas as pd

from data_process.data_processing import IMG_EXTS, load_annotations

COMBO_KEYWORD = "조합"
# zip 이름 앞의 구분자. TL_2_조합.zip → TL_2. TS/VS는 원천(이미지), TL/VL은 라벨(JSON)이다.
ZIP_KEY = re.compile(r"^([A-Z]{2}_\d+)(?:_|$)")
# 압축을 다 푼 폴더에만 남기는 표시 파일. 없으면 중간에 멈춘 것으로 보고 다시 푼다.
DONE_MARKER = ".extracted"


def _nfc(text: str) -> str:
    """Mac·Drive에서 자모가 분리된(NFD) 한글 이름을 NFC로 맞춘다."""
    return unicodedata.normalize("NFC", text)


def zip_key(path: Path) -> str | None:
    """zip 이름의 구분자(TL_2 등)를 반환한다. 형식이 다르면 None."""
    match = ZIP_KEY.match(_nfc(path.stem))
    return match.group(1) if match else None


def download_drive_folder(url: str, out_dir: Path) -> Path:
    """Google Drive 공유 폴더의 파일을 out_dir에 받고 out_dir을 반환한다.

    폴더는 "링크가 있는 모든 사용자"로 공유되어 있어야 한다. 이미 받은 파일은 건너뛰고,
    끊긴 파일은 이어 받는다. AI Hub는 해외 IP 다운로드를 막아 Colab에서는 이 방법으로 받는다.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # output 끝에 경로 구분자가 없어야 Drive 폴더 이름의 하위 폴더 없이 out_dir에 바로 받는다
    gdown.download_folder(url=url, output=str(out_dir), quiet=False, resume=True)
    return out_dir


def find_combo_zips(zip_dir: Path, exclude: Iterable[str] = ("TL_2",)) -> list[Path]:
    """zip_dir 아래에서 경로에 '조합'이 들어간 zip을 찾고, 구분자가 exclude에 있는 zip은 뺀다.

    exclude는 구분자 전체와 비교하므로 TL_2를 빼도 TL_20은 남는다.
    AI Hub에서 나눠 받은 .zip.part* 파일이 있으면 합친 뒤 실행하라고 알린다.
    """
    zip_dir = Path(zip_dir)
    if not zip_dir.is_dir():
        raise FileNotFoundError(f"zip 폴더가 없습니다: {zip_dir}")
    parts = sorted(p.name for p in zip_dir.rglob("*.zip.part*"))
    if parts:
        raise ValueError(
            f"나뉜 zip 파일이 있습니다({parts[:3]} 등). "
            "`cat 이름.zip.part* > 이름.zip`으로 합친 뒤 part 파일을 옮기고 다시 실행하세요."
        )
    exclude = set(exclude)
    zips = [
        p
        for p in zip_dir.rglob("*.zip")
        if COMBO_KEYWORD in _nfc(p.relative_to(zip_dir).as_posix()) and zip_key(p) not in exclude
    ]
    return sorted(zips, key=lambda p: _nfc(p.name))


def _member_name(info: zipfile.ZipInfo) -> str:
    """zip 안의 파일 이름. Windows에서 만든 zip(cp949, UTF-8 표시 없음)의 한글을 되살린다."""
    name = info.filename
    if not info.flag_bits & 0x800:
        try:
            name = name.encode("cp437").decode("cp949")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return _nfc(name)


def _extract(zip_path: Path, dest: Path) -> int:
    """zip을 dest에 풀고 푼 파일 수를 반환한다. dest 밖을 가리키는 항목은 거부한다."""
    root = dest.resolve()
    n_files = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = _member_name(info)
            if name.startswith("__MACOSX/") or Path(name).name.startswith("._"):
                continue
            target = (dest / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"{zip_path.name}: 폴더 밖을 가리키는 항목 {name}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            n_files += 1
    return n_files


def extract_zips(zip_paths: list[Path], out_dir: Path, overwrite: bool = False) -> list[Path]:
    """zip마다 out_dir/<zip 이름>/에 풀고 그 폴더 목록을 반환한다. 이미 다 푼 zip은 건너뛴다.

    임시 폴더에 푼 뒤 이름을 바꾸므로, 중간에 멈춰도 반쯤 풀린 폴더를 다 푼 것으로 보지 않는다.
    풀기 전에 남은 디스크 용량이 압축을 푼 크기보다 작으면 멈춘다.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dests = [out_dir / _nfc(p.stem) for p in zip_paths]
    todo = [
        (zp, dest)
        for zp, dest in zip(zip_paths, dests, strict=True)
        if overwrite or not (dest / DONE_MARKER).exists()
    ]

    need = 0
    for zp, _ in todo:
        with zipfile.ZipFile(zp) as zf:
            need += sum(info.file_size for info in zf.infolist())
    free = shutil.disk_usage(out_dir).free
    if need > free:
        raise OSError(f"디스크 용량 부족: 필요 {need / 1e9:.1f}GB, 남음 {free / 1e9:.1f}GB")

    for zp, dest in zip(zip_paths, dests, strict=True):
        if (zp, dest) not in todo:
            print(f"건너뜀(이미 풂): {dest.name}")
            continue
        tmp = out_dir / f".{dest.name}.partial"
        for folder in [tmp, dest]:
            if folder.exists():
                shutil.rmtree(folder)
        n_files = _extract(zp, tmp)
        (tmp / DONE_MARKER).touch()
        tmp.rename(dest)
        print(f"풀기 완료: {dest.name} ({n_files}개 파일)")
    return dests


def load_combo_annotations(dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """푼 폴더들의 JSON을 모두 읽어 (bbox당 한 행인 ann_df, 오류 표)를 반환한다.

    load_annotations와 같은 형식이고, source_json 앞에 zip 폴더 이름을 붙이고 zip_name 열을 더한다.
    JSON이 없는 폴더(원천 이미지 zip)는 빈 표가 되어 결과에 영향이 없다.
    AI Hub 원본은 category_id가 모두 1(Drug)이라, 대회 데이터와 같이 drug_N의 숫자
    (K-001900 → 1900)로 바꾸고 category_name은 dl_name으로 둔다. 원래 값은 category_id_raw에 남긴다.
    """
    ann_dfs, error_dfs = [], []
    for folder in dirs:
        ann_df, error_df = load_annotations(folder)
        for df in [ann_df, error_df]:
            if not df.empty:
                df["source_json"] = folder.name + "/" + df["source_json"]
        ann_dfs.append(ann_df.assign(zip_name=folder.name) if not ann_df.empty else ann_df)
        error_dfs.append(error_df)
    ann_df = pd.concat(ann_dfs, ignore_index=True)
    if not ann_df.empty:
        ann_df["category_id_raw"] = ann_df["category_id"]
        drug_no = ann_df["drug_N"].astype(str).str.removeprefix("K-")
        ann_df["category_id"] = pd.to_numeric(drug_no, errors="coerce").astype("Int64")
        ann_df["category_name"] = ann_df["dl_name"]
    return ann_df, pd.concat(error_dfs, ignore_index=True).astype({"excluded": bool})


def list_combo_images(dirs: list[Path]) -> pd.DataFrame:
    """푼 폴더들의 이미지를 하위 폴더까지 찾아 (file_name, zip_name, path) 표로 반환한다."""
    rows = [
        {"file_name": _nfc(p.name), "zip_name": folder.name, "path": p}
        for folder in dirs
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith("._")
    ]
    return pd.DataFrame(rows, columns=["file_name", "zip_name", "path"])
