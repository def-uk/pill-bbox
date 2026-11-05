"""Kaggle에서 Google Drive 공유 폴더의 ZIP만 받아 로컬에서 압축을 푼다."""

import hashlib
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import gdown

BASE_REQUIRED_DIRS = ("train_images", "train_annotations", "test_images")


def _shared_zip_entries(url: str, download_dir: Path) -> list[tuple[str, Path]]:
    """공유 폴더 목록에서 ZIP 사본까지 찾고 다운로드 대상 경로를 검사한다."""
    try:
        entries = gdown.download_folder(
            url=url,
            output=str(download_dir),
            skip_download=True,
            use_cookies=False,
            timeout=60,
        )
    except gdown.DownloadError as error:
        raise RuntimeError(
            "Drive 공유 폴더 목록을 읽지 못했습니다. Kaggle Internet과 "
            "폴더의 '링크가 있는 모든 사용자 · 뷰어' 설정을 확인하세요."
        ) from error
    result = []
    seen = set()
    for entry in entries:
        relative = Path(unicodedata.normalize("NFC", entry.path))
        lower = relative.name.lower()
        if ".zip.part" in lower:
            raise ValueError(f"분할 ZIP은 먼저 하나로 합쳐야 합니다: {relative}")
        # Drive 사본 이름(TS_5_조합.zip의 사본)도 포함한다. 다른 결과 파일은 받지 않는다.
        if not re.fullmatch(r".+\.zip(?:의\s*사본(?:\s*\(\d+\))?)?", lower):
            continue
        destination = (download_dir / relative).resolve()
        if not destination.is_relative_to(download_dir.resolve()):
            raise ValueError(f"다운로드 폴더 밖을 가리키는 경로입니다: {relative}")
        if destination in seen:
            raise ValueError(f"공유 폴더의 ZIP 저장 경로가 중복됩니다: {relative}")
        seen.add(destination)
        result.append((entry.id, destination))
    if not result:
        raise FileNotFoundError(f"공유 폴더에 ZIP 파일이 없습니다: {url}")
    return sorted(result, key=lambda entry: entry[1].as_posix())


def _download_shared_zips(url: str, cache_dir: Path, refresh: bool) -> list[Path]:
    """ZIP만 이어 받고 완료 파일은 재사용하며 ZIP이 아닌 응답은 거부한다."""
    download_dir = cache_dir / "zip"
    download_dir.mkdir(parents=True, exist_ok=True)
    entries = _shared_zip_entries(url, download_dir)
    paths = []
    for index, (file_id, destination) in enumerate(entries, 1):
        paths.append(destination)
        if not refresh and destination.is_file() and zipfile.is_zipfile(destination):
            print(f"[ZIP {index}/{len(entries)}] 로컬 재사용: {destination.name}", flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[ZIP {index}/{len(entries)}] Drive → 로컬: {destination.name}", flush=True)
        try:
            gdown.download(
                id=file_id,
                output=str(destination),
                resume=not destination.exists() and not refresh,
                use_cookies=False,
                timeout=120,
                retries=2,
            )
        except gdown.DownloadError as error:
            raise RuntimeError(
                f"ZIP 다운로드 실패: {destination.name}. 공유 권한·Internet·"
                "Drive 다운로드 제한을 확인하세요. 완료본은 다음 실행에서 재사용합니다."
            ) from error
        if not zipfile.is_zipfile(destination):
            raise ValueError(f"다운로드한 파일이 ZIP이 아닙니다: {destination}")
    return paths


def prepare_shared_drive_zips(url: str, cache_dir: Path, refresh: bool = False) -> Path:
    """공유 폴더 ZIP을 로컬에 받아 압축 해제 경로를 반환한다.

    Drive 원본에는 쓰지 않는다. 폴더 목록은 매번 확인하고 완료 ZIP과 압축 해제본을 재사용한다.
    Drive에서 기존 ZIP 내용을 바꿨다면 refresh=True로 다시 받는다.
    """
    from data_process.aihub_data import extract_zips

    if not url.startswith("https://drive.google.com/drive/folders/"):
        raise ValueError(
            "Drive 폴더 공유 링크(https://drive.google.com/drive/folders/...)가 필요합니다."
        )
    cache_dir = Path(cache_dir).expanduser().resolve()
    if cache_dir.is_relative_to(Path("/kaggle/input")):
        raise ValueError("Kaggle Input은 읽기 전용입니다. 캐시는 로컬 쓰기 경로를 지정하세요.")
    cache_dir = cache_dir / hashlib.sha256(url.encode()).hexdigest()[:16]
    paths = _download_shared_zips(url, cache_dir, refresh)
    names = [unicodedata.normalize("NFC", path.stem) for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("같은 이름의 ZIP이 있어 압축 해제 폴더가 충돌합니다.")
    manifest = [
        {
            "path": path.relative_to(cache_dir).as_posix(),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]
    identity = json.dumps(manifest, sort_keys=True)
    extracted = cache_dir / "extracted" / hashlib.sha256(identity.encode()).hexdigest()[:16]
    # 공용 함수가 디스크 여유 공간·경로 이탈·CRC를 검사하고 완료 표시를 남긴다.
    extract_zips(paths, extracted)
    # 전처리가 원본 annotation으로 오인하지 않도록 JSON은 이미지·라벨 트리 밖에 둔다.
    extracted.with_suffix(".json").write_text(identity, encoding="utf-8")
    print("공유 ZIP 준비 완료:", extracted, flush=True)
    return extracted


def find_base_raw(extracted: Path) -> Path:
    """기본 ZIP의 최상위 폴더 이름에 관계없이 원본 이미지·JSON 폴더를 찾는다."""
    candidates = []
    for image_dir in sorted(extracted.rglob("train_images")):
        root = image_dir.parent
        if all(
            (root / name).is_dir() and any((root / name).iterdir()) for name in BASE_REQUIRED_DIRS
        ):
            candidates.append(root)
    if len(candidates) != 1:
        raise ValueError(
            "기본 ZIP에서 train_images·train_annotations·test_images를 갖춘 폴더가 "
            f"하나여야 합니다. 발견: {candidates}"
        )
    return candidates[0]
