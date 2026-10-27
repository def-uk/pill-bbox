import os
import shutil
from collections.abc import Sequence
from pathlib import Path


def link_images(paths: Sequence[Path], out_dir: Path) -> dict[str, int]:
    """이미지를 out_dir에 심볼릭 링크로 모으고 방식별 개수를 반환한다.

    링크를 만들 수 없으면(Windows 권한 등) 복사한다. 이미 같은 원본을 가리키는 링크나 같은 크기의
    복사본은 그대로 두고, out_dir에서 paths에 없는 파일은 지운다. data/raw는 바꾸지 않는다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {path.name for path in paths}
    for old in out_dir.iterdir():
        if old.name not in names:
            old.unlink()
    counts = {"symlink": 0, "copy": 0, "kept": 0}
    for src in paths:
        src = src.resolve()
        dst = out_dir / src.name
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            counts["kept"] += 1
            continue
        if not dst.is_symlink() and dst.is_file() and dst.stat().st_size == src.stat().st_size:
            counts["kept"] += 1
            continue
        dst.unlink(missing_ok=True)
        try:
            dst.symlink_to(src)
            counts["symlink"] += 1
        except OSError:
            shutil.copy2(src, dst)
            counts["copy"] += 1
    return counts
