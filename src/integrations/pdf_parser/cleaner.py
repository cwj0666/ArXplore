from __future__ import annotations

import re
from typing import Any


class TextCleanerMixin:
    """cleaner mixin logic."""

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = text.replace('\x00', ' ')
        normalized = re.sub('(?<=\\w)-\\n(?=\\w)', '', normalized)
        normalized = re.sub('\\n\\d+•[A-Z][^\\n]{0,100}\\n', '\n', normalized)
        normalized = re.sub('\\n[A-Z][^\\n]{0,100}•\\d+\\n', '\n', normalized)
        normalized = re.sub("\\n[A-Z][A-Za-z0-9][A-Za-z0-9 :,\\-–'.]{3,80}\\s+\\d{1,3}\\n", '\n', normalized)
        normalized = re.sub('(?<=[a-z])(?=[A-Z][A-Za-z-]{2,})', ' ', normalized)
        normalized = re.sub('\\bar Xiv\\b', 'arXiv', normalized)
        normalized = re.sub('\\bLi DAR\\b', 'LiDAR', normalized)
        normalized = re.sub('\\bGit Hub\\b', 'GitHub', normalized)
        normalized = re.sub('\\bHugging Face\\b', 'HuggingFace', normalized)
        normalized = re.sub('\\bModel Scope\\b', 'ModelScope', normalized)
        normalized = re.sub('\\bDeep Seek\\b', 'DeepSeek', normalized)
        normalized = re.sub('[ \\t]+', ' ', normalized)
        normalized = re.sub('\\n{3,}', '\n\n', normalized)
        return normalized.strip()

    @classmethod
    def _normalize_extracted_page_text(cls, text: str) -> str:
        lines = [line.strip() for line in text.replace('\x00', ' ').splitlines()]
        filtered_lines: list[str] = []
        for line in lines:
            if not line:
                filtered_lines.append('')
                continue
            if re.fullmatch('\\d+', line):
                continue
            if re.fullmatch('\\d{4}-\\d{2}-\\d{2}', line):
                continue
            if re.match('^arXiv:\\d{4}\\.\\d{4,5}v?\\d*', line):
                continue
            if re.match('^\\*?Equal contribution', line, re.IGNORECASE):
                continue
            if re.match('^(?:/uni\\d{8})+$', line):
                continue
            if len(re.findall('/uni\\d{8}', line)) >= 3:
                continue
            if cls._looks_like_running_header_footer(line):
                continue
            if cls._looks_like_toc_line(line):
                continue
            filtered_lines.extend(cls._split_inline_heading_line(line))
        stitched_lines: list[str] = []
        for line in filtered_lines:
            if not line:
                if stitched_lines and stitched_lines[-1] != '':
                    stitched_lines.append('')
                continue
            if stitched_lines:
                previous = stitched_lines[-1]
                if previous and cls._should_merge_lines(previous, line):
                    stitched_lines[-1] = cls._merge_lines(previous, line)
                    continue
            stitched_lines.append(line)
        return '\n'.join(stitched_lines).strip()

    @staticmethod
    def _normalize_layout_heading_like_text(text: str) -> str:
        normalized = ' '.join(text.split())
        prefix = ''
        tokens = normalized.split()
        if len(tokens) >= 3 and re.fullmatch('[A-Z](?:\\.\\d+)*', tokens[0]) and re.fullmatch('[A-Z]', tokens[1]) and re.fullmatch('[A-Z]{2,}', tokens[2]):
            prefix = tokens[0]
            normalized = ' '.join(tokens[1:])
        previous = None
        while previous != normalized:
            previous = normalized
            normalized = re.sub('\\b([A-Z])\\s+([A-Z]{2,})\\b', lambda match: f'{match.group(1)}{match.group(2)}', normalized)
            normalized = re.sub('\\b([A-Z])\\s+([A-Z][a-z][A-Za-z-]*)\\b', lambda match: f'{match.group(1)}{match.group(2)[0].lower()}{match.group(2)[1:]}', normalized)
        normalized = re.sub('\\s+([:;,.!?])', '\\1', normalized)
        if prefix:
            normalized = f'{prefix} {normalized}'
        return normalized.strip()

    @staticmethod
    def _looks_like_numbered_heading(title: str) -> bool:
        words = re.findall('[A-Za-z0-9-]+', title)
        if not words:
            return False
        uppercase_like = sum(1 for word in words if word[0].isupper() or word[0].isdigit() or word.isupper())
        ratio = uppercase_like / len(words)
        if len(words) <= 4:
            return ratio >= 0.5
        return ratio >= 0.6

    @classmethod
    def _prettify_section_title(cls, title: str) -> str:
        compact = ' '.join(title.split())
        if compact.isupper():
            return re.sub('[A-Z]+', lambda match: match.group(0) if match.group(0) in cls._KNOWN_UPPERCASE_TITLE_TOKENS or len(match.group(0)) == 1 else match.group(0).capitalize(), compact)
        return compact

    @classmethod
    def _strip_inline_heading_prefix(cls, text: str) -> str:
        split = cls._find_inline_heading_split(text)
        if split is not None:
            (_, rest) = split
            return rest
        return text

    @staticmethod
    def _normalize_chunk_opening(text: str) -> str:
        compact = text.lstrip()
        if not compact:
            return compact
        bullet_index = compact.find('• ')
        if bullet_index != -1 and bullet_index <= 220:
            prefix = compact[:bullet_index]
            if re.fullmatch("[\\s,.;:()\\[\\]0-9A-Za-z&'\\-–/]+", prefix or ' '):
                compact = compact[bullet_index + 2:].lstrip()
        compact = re.sub('^[,;:)\\]\\}]+\\s*', '', compact)
        return compact

    @classmethod
    def _split_inline_heading_line(cls, line: str) -> list[str]:
        line = line.strip()
        if not line:
            return ['']
        split = cls._find_inline_heading_split(line)
        if split is not None:
            (head, rest) = split
            return [head, rest]
        return [line]

    @classmethod
    def _find_inline_heading_split(cls, text: str) -> tuple[str, str] | None:
        compact = ' '.join(text.split())
        title_prefix = cls._find_short_title_prefix(compact)
        if title_prefix is not None:
            return title_prefix
        if not re.match('^\\d+(?:\\.\\d+)+(?:[.)])?\\s+', compact):
            return None
        for starter in cls._INLINE_BODY_STARTERS:
            marker = f' {starter} '
            index = compact.find(marker)
            if index == -1:
                continue
            head = compact[:index].strip()
            rest = compact[index + 1:].strip()
            if len(head.split()) > 16 or len(head) < 8:
                continue
            if not rest:
                continue
            return (head, rest)
        return None

    @staticmethod
    def _find_short_title_prefix(text: str) -> tuple[str, str] | None:
        match = re.match("^(?P<head>[A-Z0-9][A-Za-z0-9/+&' -]{2,48}\\.)\\s*(?P<rest>[A-Z].+)$", text)
        if match is None:
            return None
        head = match.group('head').strip()
        rest = match.group('rest').strip()
        head_words = head[:-1].split()
        if not 1 <= len(head_words) <= 6:
            return None
        if head.startswith(('This ', 'These ', 'We ', 'Our ', 'In ', 'To ', 'As ', 'However ')):
            return None
        if any(len(word) > 20 for word in head_words):
            return None
        return (head, rest)

    @staticmethod
    def _looks_like_running_header_footer(line: str) -> bool:
        compact = ' '.join(line.split())
        if re.search('•\\d{1,3}$', compact):
            return True
        if re.fullmatch("\\d+•[A-Z][A-Za-z .'-]+", compact):
            return True
        if re.fullmatch("[A-Z][A-Za-z0-9:,\\-–' ]{8,100}\\d{1,3}$", compact):
            words = compact[:-3].split()
            if 2 <= len(words) <= 16:
                return True
        return False

    @staticmethod
    def _looks_like_toc_line(line: str) -> bool:
        compact = ' '.join(line.split())
        if compact.lower() in {'contents', 'table of contents'}:
            return True
        if re.search('(?:\\. ?){6,}\\d+$', compact):
            return True
        if re.fullmatch('(?:\\d+(?:\\.\\d+)*\\.?)\\s+.+(?:\\. ?){3,}\\d{1,3}', compact):
            return True
        if re.fullmatch("(?:\\d+(?:\\.\\d+)*|[A-Z])\\s+[A-Z][A-Za-z0-9 ,:/()'&-]{1,80}\\s+\\d{1,3}", compact):
            return True
        return bool(re.fullmatch('(?:\\d+(?:\\.\\d+)*)\\s+.+\\s+\\d{1,3}', compact) and compact.count('.') >= 4)

    @staticmethod
    def _starts_like_body_paragraph(first_line: str, compact: str) -> bool:
        normalized_first_line = ' '.join(first_line.split())
        if not normalized_first_line:
            return False
        if re.match('^(?:As shown|Beyond|In this|In our|We |To address|To evaluate|To further|Notably|While|Since|Early|Figure \\d+ (?:shows|illustrates|visualizes)|Table \\d+ (?:presents|shows|compares|reports|summarizes))\\b', normalized_first_line, re.IGNORECASE):
            return True
        word_count = len(re.findall('[A-Za-z]+', normalized_first_line))
        numeric_cells = len(re.findall('\\b\\d+(?:\\.\\d+)?\\b', normalized_first_line))
        if word_count >= 12 and numeric_cells <= 3 and re.match('^[A-Z][a-z]', normalized_first_line):
            return True
        compact_prefix = compact[:220]
        if compact_prefix.count('. ') >= 1 and word_count >= 10 and (numeric_cells <= 4):
            return True
        return False

    @classmethod
    def _should_merge_lines(cls, previous: str, current: str) -> bool:
        if not previous or not current:
            return False
        if previous.endswith('-'):
            return True
        if previous.endswith(('.', '?', '!', ':', ';')):
            return False
        if cls._normalize_section_heading(current) is not None:
            return False
        if current[0].islower() or current[0] in '(["\'':
            return True
        if previous[-1] in ',/':
            return True
        return len(previous) > 40 and current[:1].isupper() and (len(current.split()) > 3)

    @staticmethod
    def _merge_lines(previous: str, current: str) -> str:
        if previous.endswith('-'):
            return previous[:-1] + current.lstrip()
        return previous.rstrip() + ' ' + current.lstrip()

    @classmethod
    def _should_drop_section(cls, title: str, text: str) -> bool:
        compact = ' '.join(text.split())
        if not compact:
            return True
        if cls._infer_content_role(title) == 'toc':
            return True
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines and len(lines) <= 12 and all(cls._looks_like_toc_line(line) for line in lines):
            return True
        return False

    @classmethod
    def _strip_trailing_reference_like_tail(cls, title: str, text: str) -> str:
        lowered_title = title.lower()
        if not any(keyword in lowered_title for keyword in ('conclusion', 'discussion', 'appendix', 'additional analysis', 'supplementary', 'limitations', 'experimental details', 'implementation details')):
            return text
        paragraphs = [paragraph.strip() for paragraph in re.split('\\n{2,}', text) if paragraph.strip()]
        if not paragraphs:
            return text
        while paragraphs and cls._looks_like_reference_tail_paragraph(paragraphs[-1]):
            paragraphs.pop()
        trimmed = '\n\n'.join(paragraphs).strip()
        return trimmed or text

    @staticmethod
    def _looks_like_reference_tail_paragraph(text: str) -> bool:
        compact = ' '.join(text.split())
        if not compact:
            return False
        reference_markers = len(re.findall('\\[\\d+\\]', compact))
        year_markers = len(re.findall('\\b(?:19|20)\\d{2}\\b', compact))
        url_markers = len(re.findall('https?\\s*:\\s*//|doi:|arxiv:', compact, re.IGNORECASE))
        sentence_breaks = compact.count('. ') + compact.count('? ') + compact.count('! ')
        venue_markers = len(re.findall('\\b(?:Proceedings|Conference|CVPR|ICCV|ECCV|NeurIPS|ICLR|ACL|EMNLP|AAAI|Google)\\b', compact, re.IGNORECASE))
        author_list_like = bool(re.match("^(?:\\[\\d+\\]\\s*)?[A-Z][A-Za-z'`.-]+,\\s+[A-Z](?:\\.[A-Z])?(?:,\\s+[A-Z][A-Za-z'`.-]+,\\s+[A-Z](?:\\.[A-Z])?){1,}", compact))
        if reference_markers >= 2:
            return True
        if reference_markers >= 1 and year_markers >= 2 and (sentence_breaks <= 3):
            return True
        if url_markers >= 1 and year_markers >= 1 and (venue_markers >= 1):
            return True
        if author_list_like and year_markers >= 1:
            return True
        if compact.startswith(('In: ', '[', 'doi:', 'https://', 'http://')) and (year_markers >= 1 or venue_markers >= 1):
            return True
        return False

    @classmethod
    def _normalize_section_heading(cls, line: str) -> str | None:
        candidate = cls._normalize_layout_heading_like_text(' '.join(line.split()))
        candidate = re.sub('\\s*[–—]\\s*', ' - ', candidate)
        lowered = candidate.lower()
        if lowered in {'figure', 'table', 'listing'}:
            return None
        if lowered.startswith(('figure ', 'table ', 'listing ', 'arxiv:')):
            return None
        if re.fullmatch('\\d+', candidate):
            return None
        if lowered in cls._KNOWN_SECTION_TITLES:
            return cls._prettify_section_title(candidate)
        appendix_match = re.fullmatch("(appendix(?:\\s+[A-Z])?)\\s+([A-Z][A-Za-z0-9 ,:/()'&-]{1,100})", candidate, re.IGNORECASE)
        if appendix_match:
            prefix = appendix_match.group(1)
            title = appendix_match.group(2)
            if title.lower() in {'table of contents', 'contents'}:
                return cls._prettify_section_title(prefix)
            return cls._prettify_section_title(f'{prefix} {title}')
        match = cls._NUMBERED_SECTION_PATTERN.fullmatch(candidate)
        if match is None:
            return None
        prefix = match.group('prefix')
        title = match.group('title')
        prefix_root = prefix.split('.')[0]
        if prefix_root.isdigit():
            major_prefix = int(prefix_root)
            if major_prefix >= 20:
                return None
            if prefix.isdigit() and int(prefix) >= 50:
                return None
        if re.search('(?:\\. ?){2,}\\d{1,3}$', candidate):
            return None
        if re.search('\\s\\d{1,3}$', title):
            return None
        if title.lower().startswith(('figure ', 'table ', 'listing ')):
            return None
        title_words = title.split()
        if len(title_words) > 9:
            return None
        if any(len(word) > 24 for word in title_words):
            return None
        if ',' in title and len(title_words) > 6:
            return None
        if title.startswith(('We ', 'Our ', 'This ', 'These ')):
            return None
        if not cls._looks_like_numbered_heading(title):
            return None
        return cls._prettify_section_title(f'{prefix} {title}')

    @classmethod
    def _normalize_section_text(cls, title: str, text: str) -> str:
        compact = cls._normalize_text(text)
        if not compact:
            return compact
        compact = cls._strip_trailing_reference_like_tail(title, compact)
        if cls._infer_content_role(title) == 'body':
            lead_fragment_match = re.match('^(?P<lead>[a-z][^.]{0,160}\\.)\\s+(?P<rest>(?:In this section|Here, we|We |This section|To )\\b.+)$', compact)
            if lead_fragment_match:
                return lead_fragment_match.group('rest').strip()
        return compact

    @classmethod
    def _extract_sections(cls, text: str) -> list[dict[str, Any]]:
        lines = [line.strip() for line in text.splitlines()]
        sections: list[dict[str, Any]] = []
        current_title = 'Front Matter'
        current_lines: list[str] = []
        for line in lines:
            if not line:
                if current_lines and current_lines[-1] != '':
                    current_lines.append('')
                continue
            heading = cls._normalize_section_heading(line)
            if heading is not None:
                if current_lines:
                    section_text = cls._normalize_text('\n'.join(current_lines))
                    section_text = cls._normalize_section_text(current_title, section_text)
                    if section_text and (not cls._should_drop_section(current_title, section_text)):
                        sections.append({'title': current_title, 'text': section_text})
                current_title = heading
                current_lines = []
                continue
            current_lines.append(line)
        if current_lines:
            section_text = cls._normalize_text('\n'.join(current_lines))
            section_text = cls._normalize_section_text(current_title, section_text)
            if section_text and (not cls._should_drop_section(current_title, section_text)):
                sections.append({'title': current_title, 'text': section_text})
        if len(sections) == 1 and sections[0]['title'] == 'Front Matter':
            sections[0]['title'] = 'Full Text'
        return cls._reorder_sections(sections)

    @classmethod
    def _reorder_sections(cls, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(sections) < 3:
            return sections
        indexed_sections = list(enumerate(sections))
        numeric_sections = [(index, sort_key) for (index, section) in indexed_sections if (sort_key := cls._parse_numeric_section_sort_key(str(section.get('title') or ''))) is not None]
        if len(numeric_sections) < 2:
            return sections
        numeric_order = [sort_key for (_, sort_key) in numeric_sections]
        if numeric_order == sorted(numeric_order):
            return sections

        def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, tuple[int, ...], int]:
            (index, section) = item
            title = str(section.get('title') or '')
            lowered = title.lower()
            if lowered == 'front matter':
                return (0, (), index)
            if lowered == 'abstract':
                return (1, (), index)
            numeric_key = cls._parse_numeric_section_sort_key(title)
            if numeric_key is not None:
                return (2, numeric_key, index)
            return (3, (), index)
        return [section for (_, section) in sorted(indexed_sections, key=sort_key)]

    @staticmethod
    def _parse_numeric_section_sort_key(title: str) -> tuple[int, ...] | None:
        match = re.match('^(?P<prefix>\\d+(?:\\.\\d+)*)\\b', title.strip())
        if match is None:
            return None
        return tuple(int(part) for part in match.group('prefix').split('.'))

