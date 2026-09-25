#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKLOG = ROOT / "docs" / "worklog"
INDEX_NAME = "INDEX.md"
DIGEST_NAME = "TROUBLESHOOTING.md"
GENERATED_NAMES = frozenset({INDEX_NAME, DIGEST_NAME})
AREAS = ("pipeline", "retrieval", "agent", "backend", "frontend", "eval", "infra", "docs", "tooling")
REQUIRED_KEYS = ("title", "date", "area", "decision")
TROUBLE_SECTION = "**트러블슈팅**"
SECTIONS = ("**결정과 근거**", "**트레이드오프**", "**eval 영향**", "**알려진 한계**", TROUBLE_SECTION)
CASE_LABELS = ("증상", "원인", "확인 방법", "해결", "재발 방지")
CASE_LABEL_RE = re.compile(r"^- (" + "|".join(re.escape(label) for label in CASE_LABELS) + r"):")
NOT_APPLICABLE = "해당 없음"
ENTRY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{2})_[^/]+\.md$")
PHASE_RE = re.compile(r"^phase-(\d+)$")
GENERATOR_HINT = "`python scripts/worklog.py index`가 생성한다. 손으로 고치지 않는다."
TEMPLATE = """---
title: {title}
date: {date}
area: [{area}]
decision: (한 줄. 색인에 그대로 실린다)
---

**결정과 근거** — 무엇을 어떻게 하기로 했고, 왜 그런가. 근거가 실측이면 방법·결과를 포함한다.

**트레이드오프** — 고려한 대안 / 채택 이유 / 지불한 비용. 미룬 것·좁힌 것·복잡해진 것을 적는다.

**eval 영향** — 이 결정이 만드는(또는 없애는) 측정 지점. 없으면 "해당 없음".

**알려진 한계** — 이 결정으로 못 잡게 된 것. 없으면 "해당 없음".

**트러블슈팅** — 해당 없음
"""


class Problem(Exception):
    pass


def parse_frontmatter(text: str, path: Path) -> dict[str, object]:
    """`---`로 감싼 frontmatter를 `key: value` 줄 단위로 읽는다. `[a, b]` 값은 목록이 된다."""
    if not text.startswith("---\n"):
        raise Problem(f"{path}: frontmatter(---)로 시작해야 한다")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise Problem(f"{path}: frontmatter 닫는 --- 없음")
    meta: dict[str, object] = {}
    for line in text[4:end].splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise Problem(f"{path}: frontmatter 줄 형식 오류: {line!r}")
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key.strip()] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key.strip()] = value
    return meta


def section_start(text: str, section: str) -> int:
    """줄 머리에 놓인 섹션 제목의 위치. 없으면 -1."""
    m = re.search(r"^" + re.escape(section), text, re.MULTILINE)
    return m.start() if m else -1


def trouble_section(text: str) -> str:
    """본문에서 트러블슈팅 절(제목 줄부터 파일 끝까지)을 앞뒤 공백 없이 돌려준다."""
    return text[section_start(text, TROUBLE_SECTION) :].strip()


def trouble_cases(section: str, path: Path) -> int:
    """트러블슈팅 절의 사례 수. 해당 없음이면 0이고, 사례는 다섯 라벨 줄을 정해진 순서로 가져야 한다."""
    body = section[len(TROUBLE_SECTION) :].strip().lstrip("—").strip()
    if body.startswith(NOT_APPLICABLE):
        return 0
    labels = [m.group(1) for line in section.splitlines() if (m := CASE_LABEL_RE.match(line))]
    if not labels:
        raise Problem(f"{path}: 트러블슈팅 절은 '{NOT_APPLICABLE}'이거나 '- 증상:'부터 시작하는 사례를 가져야 한다")
    if len(labels) % len(CASE_LABELS) or any(
        label != CASE_LABELS[i % len(CASE_LABELS)] for i, label in enumerate(labels)
    ):
        order = " → ".join(CASE_LABELS)
        raise Problem(f"{path}: 트러블슈팅 사례의 라벨 줄은 '- 라벨:' 형식으로 {order} 순서를 반복해야 한다")
    return len(labels) // len(CASE_LABELS)


def load_entry(path: Path, phase: str) -> dict[str, object]:
    """항목 파일 하나를 검사하고 색인·모음에 쓸 필드를 돌려준다. 형식이 어긋나면 Problem을 던진다."""
    m = ENTRY_RE.match(path.name)
    if not m:
        raise Problem(f"{path}: 파일명은 YYYY-MM-DD_NN_제목.md 형식이어야 한다")
    text = path.read_text(encoding="utf-8")
    meta = parse_frontmatter(text, path)
    missing = [k for k in REQUIRED_KEYS if k not in meta]
    if missing:
        raise Problem(f"{path}: frontmatter에 {', '.join(missing)} 없음")
    if meta["date"] != m.group(1):
        raise Problem(f"{path}: frontmatter date({meta['date']})가 파일명 날짜({m.group(1)})와 다르다")
    areas = meta["area"] if isinstance(meta["area"], list) else [str(meta["area"])]
    bad = [a for a in areas if a not in AREAS]
    if bad:
        raise Problem(f"{path}: 허용되지 않는 area {bad}. 허용: {', '.join(AREAS)}")
    positions = [section_start(text, s) for s in SECTIONS]
    absent = [s for s, pos in zip(SECTIONS, positions, strict=True) if pos < 0]
    if absent:
        raise Problem(f"{path}: 필수 섹션 없음(줄 머리에 있어야 한다): {', '.join(absent)}")
    if positions != sorted(positions):
        raise Problem(f"{path}: 섹션 순서는 {' / '.join(SECTIONS)}이어야 한다")
    section = trouble_section(text)
    return {
        "path": path,
        "phase": phase,
        "date": m.group(1),
        "seq": m.group(2),
        "title": meta["title"],
        "areas": areas,
        "decision": meta["decision"],
        "trouble": section,
        "cases": trouble_cases(section, path),
    }


def phase_dirs() -> list[Path]:
    return sorted(
        (p for p in WORKLOG.iterdir() if p.is_dir() and PHASE_RE.match(p.name)),
        key=lambda p: int(PHASE_RE.match(p.name).group(1)),
    )


def newest_first(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(entries, key=lambda e: (e["date"], e["seq"]), reverse=True)


def render_index(phase_dir: Path, entries: list[dict[str, object]]) -> str:
    """페이즈 디렉터리의 INDEX.md 본문. 항목은 최신 순이다."""
    n = PHASE_RE.match(phase_dir.name).group(1)
    lines = [
        f"# Phase {n} 작업 로그 색인",
        "",
        GENERATOR_HINT,
        "항목을 찾을 때는 이 표만 읽고 필요한 파일만 연다.",
        "",
        "| 날짜 | 제목 | 영역 | 결정 | 파일 |",
        "|---|---|---|---|---|",
    ]
    for e in newest_first(entries):
        name = e["path"].name
        lines.append(f"| {e['date']} | {e['title']} | {', '.join(e['areas'])} | {e['decision']} | [{name}]({name}) |")
    return "\n".join(lines) + "\n"


def entry_link(entry: dict[str, object]) -> str:
    return f"[{entry['title']}](phase-{entry['phase']}/{entry['path'].name})"


def render_digest(by_phase: list[tuple[str, list[dict[str, object]]]]) -> str:
    """트러블슈팅 사례가 있는 항목만 모은 TROUBLESHOOTING.md 본문. 페이즈는 최신 순, 페이즈 안의 항목도 최신 순이다."""
    groups = [(phase, newest_first([e for e in entries if e["cases"]])) for phase, entries in reversed(by_phase)]
    groups = [(phase, entries) for phase, entries in groups if entries]
    total_entries = sum(len(entries) for _, entries in groups)
    total_cases = sum(e["cases"] for _, entries in groups for e in entries)
    lines = [
        "# 트러블슈팅 모음",
        "",
        GENERATOR_HINT,
        "각 항목의 **트러블슈팅** 절을 그대로 옮긴다. 고칠 때는 항목 파일을 고치고 `index`를 다시 실행한다.",
        "",
        f"항목 {total_entries}개, 사례 {total_cases}건.",
        "",
        "| phase | 날짜 | 제목 | 영역 |",
        "|---|---|---|---|",
    ]
    for phase, entries in groups:
        for e in entries:
            lines.append(f"| {phase} | {e['date']} | {entry_link(e)} | {', '.join(e['areas'])} |")
    for phase, entries in groups:
        lines.extend(["", f"## Phase {phase}"])
        for e in entries:
            lines.extend(
                [
                    "",
                    f"### {entry_link(e)}",
                    "",
                    f"phase {phase} · {e['date']} · {', '.join(e['areas'])}",
                    "",
                    e["trouble"],
                ]
            )
    return "\n".join(lines) + "\n"


def collect() -> tuple[dict[Path, str], list[str]]:
    """모든 항목을 검사해 생성할 파일(INDEX.md 각각과 TROUBLESHOOTING.md)의 내용과 문제 목록을 돌려준다."""
    rendered: dict[Path, str] = {}
    problems: list[str] = []
    by_phase: list[tuple[str, list[dict[str, object]]]] = []
    for legacy in WORKLOG.glob("phase-*.md"):
        problems.append(
            f"{legacy}: 페이즈 단일 파일은 쓰지 않는다. docs/worklog/{legacy.stem}/ 아래 항목 파일로 나눈다"
        )
    for phase_dir in phase_dirs():
        phase = PHASE_RE.match(phase_dir.name).group(1)
        entries = []
        for path in sorted(phase_dir.glob("*.md")):
            if path.name in GENERATED_NAMES:
                continue
            try:
                entries.append(load_entry(path, phase))
            except Problem as exc:
                problems.append(str(exc))
        rendered[phase_dir / INDEX_NAME] = render_index(phase_dir, entries)
        by_phase.append((phase, entries))
    rendered[WORKLOG / DIGEST_NAME] = render_digest(by_phase)
    return rendered, problems


def cmd_index(check: bool) -> int:
    rendered, problems = collect()
    for path, text in rendered.items():
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != text:
            rel = path.relative_to(ROOT)
            if check:
                problems.append(f"{rel}: 생성 파일이 항목과 다르다. `python scripts/worklog.py index` 실행")
            else:
                path.write_text(text, encoding="utf-8")
                print(f"wrote {rel}")
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


def cmd_new(phase: int, title: str, area: str) -> int:
    areas = [a.strip() for a in area.split(",") if a.strip()]
    bad = [a for a in areas if a not in AREAS]
    if bad:
        print(f"허용되지 않는 area {bad}. 허용: {', '.join(AREAS)}", file=sys.stderr)
        return 1
    phase_dir = WORKLOG / f"phase-{phase}"
    phase_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    taken = [ENTRY_RE.match(p.name).group(2) for p in phase_dir.glob(f"{today}_*.md") if ENTRY_RE.match(p.name)]
    seq = f"{max((int(s) for s in taken), default=0) + 1:02d}"
    slug = re.sub(r"[\s/\\:*?\"<>|()\[\]]+", "-", title.strip()).strip("-")
    path = phase_dir / f"{today}_{seq}_{slug}.md"
    if path.exists():
        print(f"이미 있음: {path}", file=sys.stderr)
        return 1
    path.write_text(TEMPLATE.format(title=title.strip(), date=today, area=", ".join(areas)), encoding="utf-8")
    print(path.relative_to(ROOT))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="worklog 항목 생성·색인과 트러블슈팅 모음 생성·검사")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="phase-*/INDEX.md와 TROUBLESHOOTING.md 재생성")
    sub.add_parser("check", help="항목 형식과 생성 파일 최신 여부 검사 (CI·훅)")
    new = sub.add_parser("new", help="오늘 날짜로 항목 파일 생성")
    new.add_argument("title")
    new.add_argument("--phase", type=int, required=True)
    new.add_argument("--area", required=True, help=f"쉼표 구분. 허용: {', '.join(AREAS)}")
    args = parser.parse_args(argv)
    if args.cmd == "index":
        return cmd_index(check=False)
    if args.cmd == "check":
        return cmd_index(check=True)
    return cmd_new(args.phase, args.title, args.area)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
