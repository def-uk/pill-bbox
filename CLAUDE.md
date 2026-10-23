# CLAUDE.md

이 저장소의 에이전트 지침은 `AGENTS.md` 한 곳에서 관리합니다(Codex와 공유).
규칙을 추가하거나 바꿀 때는 이 파일이 아니라 `AGENTS.md`를 고치세요.
이 파일에는 Claude Code에만 해당하는 내용만 둡니다.

@AGENTS.md

## Claude Code 전용

- 개인 설정은 `CLAUDE.local.md`에 둡니다(커밋되지 않음). 이 파일이 `AGENTS.md`를 불러오므로 `CLAUDE.local.md`가 있어도 공용 지침은 그대로 적용됩니다.
- md 문서 잠금(`AGENTS.md`의 "md 문서 잠금")은 `.claude/settings.json`의 PreToolUse hook이 강제합니다. 잠긴 동안 `.md` 파일에 대한 Edit·Write와 md를 고치는 Bash 명령은 거부되고, `unlock` 실행은 잠금 상태와 관계없이 항상 거부됩니다. 거부되면 우회하지 말고 사용자에게 아래 명령을 터미널에 입력해 달라고 요청합니다. Claude Code 입력창에서 `!`를 앞에 붙여 실행해도 됩니다.
  - 해제: `uv run python scripts/md_lock.py unlock`
  - md 수정을 마친 뒤 다시 잠그기: `uv run python scripts/md_lock.py lock`
