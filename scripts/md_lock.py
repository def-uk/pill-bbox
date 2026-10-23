"""md 문서 잠금: AI 에이전트가 사용자 허락 없이 저장소의 .md 파일을 고치지 못하게 막는다.

사용법 (프로젝트 루트에서, 사람이 직접 실행):
    uv run python scripts/md_lock.py unlock   # md를 고칠 동안만 잠금 해제
    uv run python scripts/md_lock.py lock     # 다시 잠그기
    uv run python scripts/md_lock.py status   # 현재 상태 확인

잠금은 두 겹이다.
1. md 파일을 읽기 전용으로 바꾼다. Codex 등 모든 에이전트의 파일 쓰기가 실패한다.
2. Claude Code는 .claude/settings.json의 PreToolUse hook(`hook` 하위 명령)이 md 수정을 거부한다.

`.md_unlocked` 파일이 없으면 잠긴 상태로 본다. 새로 clone한 직후에도 hook은 잠금으로 동작한다.
실수를 막는 장치일 뿐 보안 장치는 아니다.
"""

import json
import re
import stat
import subprocess
import sys
from pathlib import Path

# hook으로 자주 실행되므로 torch를 불러오는 src.config 대신 같은 방식으로 직접 계산한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNLOCK_MARKER = PROJECT_ROOT / ".md_unlocked"
LOCK_CMD = "uv run python scripts/md_lock.py lock"
UNLOCK_CMD = "uv run python scripts/md_lock.py unlock"
WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH

# 잠긴 동안 Bash 명령에서 막을 패턴. .md 경로와 함께 쓰일 때만 막는다.
MD_PATH = re.compile(r"\.md\b", re.IGNORECASE)
MD_WRITE_PATTERNS = [
    re.compile(r"\b(sed|perl)\b[^|;&]*\s-i"),
    re.compile(r"\b(chmod|chflags|chattr|attrib|truncate|tee|mv|cp|rm|touch|ln)\b"),
    re.compile(r">>?\s*['\"]?[^\s'\"]*\.md\b", re.IGNORECASE),
    re.compile(r"\bgit\s+(checkout|restore|apply|mv|rm|stash)\b"),
]
UNLOCK_CALL = re.compile(r"md_lock\.py\b.*\bunlock\b|\.md_unlocked")


def is_locked() -> bool:
    """잠금 상태이면 True. 해제 표시 파일이 없으면 잠긴 것으로 본다."""
    return not UNLOCK_MARKER.exists()


def list_md_files() -> list[Path]:
    """git이 관리하거나 무시하지 않는 저장소 안의 .md 파일 목록."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.md"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    paths = {PROJECT_ROOT / p for p in out.split("\0") if p}
    return sorted(p for p in paths if p.is_file())


def set_writable(path: Path, writable: bool) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IWUSR if writable else mode & ~WRITE_BITS)


def lock() -> None:
    UNLOCK_MARKER.unlink(missing_ok=True)
    files = list_md_files()
    for path in files:
        set_writable(path, False)
    print(f"md 잠금: {len(files)}개 파일을 읽기 전용으로 바꿨습니다.")
    print(f"고칠 일이 생기면: {UNLOCK_CMD}")


def unlock() -> None:
    UNLOCK_MARKER.touch()
    files = list_md_files()
    for path in files:
        set_writable(path, True)
    print(f"md 잠금 해제: {len(files)}개 파일을 쓸 수 있습니다.")
    print(f"수정이 끝나면 다시 잠그세요: {LOCK_CMD}")


def status() -> None:
    files = list_md_files()
    writable = [p for p in files if p.stat().st_mode & stat.S_IWUSR]
    if is_locked():
        print(f"잠김. md {len(files)}개 중 쓰기 가능 {len(writable)}개.")
        if writable:
            print(f"읽기 전용으로 맞추려면: {LOCK_CMD}")
    else:
        print(f"해제됨. 수정이 끝나면: {LOCK_CMD}")


def is_repo_md(file_path: str) -> bool:
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    return (
        path.suffix.lower() == ".md"
        and path.is_relative_to(PROJECT_ROOT)
        and ".venv" not in path.relative_to(PROJECT_ROOT).parts
    )


def deny(reason: str) -> None:
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }
    print(json.dumps(output, ensure_ascii=False))


def hook() -> None:
    """Claude Code PreToolUse hook. 잠긴 동안 md 수정과 잠금 해제 시도를 거부한다."""
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command") or ""

    # 잠금 해제는 상태와 관계없이 사용자만 실행한다.
    if UNLOCK_CALL.search(command):
        deny(f"md 잠금 해제는 사용자만 합니다. 사용자에게 `{UNLOCK_CMD}` 실행을 요청하세요.")
        return
    if not is_locked():
        return

    reason = (
        "md 파일이 잠겨 있습니다. 고칠 파일과 이유를 알리고, 사용자에게 터미널에서 "
        f"`{UNLOCK_CMD}`를 실행해 달라고 요청하세요. 수정이 끝나면 `{LOCK_CMD}`를 안내하세요."
    )
    file_path = tool_input.get("file_path")
    if file_path and is_repo_md(file_path):
        deny(reason)
    elif MD_PATH.search(command) and any(p.search(command) for p in MD_WRITE_PATTERNS):
        deny(reason)


COMMANDS = {"lock": lock, "unlock": unlock, "status": status, "hook": hook}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(f"사용법: uv run python scripts/md_lock.py {{{'|'.join(COMMANDS)}}}", file=sys.stderr)
        sys.exit(2)
    COMMANDS[sys.argv[1]]()
