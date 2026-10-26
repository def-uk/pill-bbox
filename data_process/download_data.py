import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import gdown

from src.config import PROJECT_ROOT

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_data(
    url: str | None = None, output_dir: Path = RAW_DATA_DIR, overwrite: bool = False
) -> Path:
    """Google Drive 공유 링크의 zip을 받아 output_dir에 풀고, output_dir을 반환한다.

    url을 넘기지 않으면 환경 변수(.env) DATA_URL을 쓴다.
    링크는 "링크가 있는 모든 사용자"로 공유되어 있어야 한다.
    output_dir에 이미 파일이 있으면 받지 않는다. 다시 받으려면 overwrite=True.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        print(f"이미 데이터가 있어 다운로드를 건너뜁니다: {output_dir}")
        return output_dir

    url = url or os.getenv("DATA_URL")
    if not url:
        raise ValueError("데이터 링크가 없습니다. .env 또는 환경 변수에 DATA_URL을 설정하세요.")

    output_dir.mkdir(parents=True, exist_ok=True)
    # 같은 디스크에 풀어야 옮길 때 복사 없이 이름만 바뀐다
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as tmp:
        tmp_dir = Path(tmp)
        zip_path = Path(gdown.download(url, output=str(tmp_dir / "data.zip")))
        if not zipfile.is_zipfile(zip_path):
            raise ValueError("zip 파일이 아닙니다. 링크와 공유 권한을 확인하세요.")

        extract_dir = tmp_dir / "extracted"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        _move_contents(_unwrap_single_dir(extract_dir), output_dir, overwrite)
    print(f"{output_dir}: {sorted(p.name for p in output_dir.iterdir())}")
    return output_dir


def _unwrap_single_dir(extract_dir: Path) -> Path:
    """zip 안이 폴더 하나로 감싸져 있으면(raw/train_images/...) 그 폴더를 반환한다."""
    children = [p for p in extract_dir.iterdir() if p.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extract_dir


def _move_contents(src_dir: Path, dst_dir: Path, overwrite: bool) -> None:
    for src in src_dir.iterdir():
        if src.name == "__MACOSX":
            continue
        dst = dst_dir / src.name
        if dst.exists():
            if not overwrite:
                continue
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))
