from __future__ import annotations

import re
from typing import Any

from .section_roles import is_references_section_title


class SemanticChunkerMixin:
    """chunker mixin logic."""

    @classmethod
    def build_chunks(cls, text: str, *, sections: list[dict[str, Any]] | None=None, max_chars: int=1800, overlap_chars: int=200) -> list[dict[str, Any]]:
        """문장/문단 경계를 우선 고려해 청크를 생성한다."""
        normalized = cls._normalize_text(text)
        if not normalized:
            return []
        max_chars = max(300, max_chars)
        overlap_chars = max(0, min(overlap_chars, max_chars // 2))
        chunk_index = 0
        chunks: list[dict[str, Any]] = []
        normalized_sections = sections or [{'title': 'Full Text', 'text': normalized}]
        for (section_index, section) in enumerate(normalized_sections):
            section_title = str(section.get('title') or 'Full Text').strip() or 'Full Text'
            section_text = cls._normalize_text(str(section.get('text') or ''))
            if not section_text:
                continue
            start = 0
            section_chunk_index = 0
            while start < len(section_text):
                end = cls._adjust_chunk_end(section_text, start, max_chars)
                candidate = section_text[start:end]
                clean_chunk = cls._strip_inline_heading_prefix(candidate.strip())
                clean_chunk = cls._normalize_chunk_opening(clean_chunk)
                if clean_chunk:
                    metadata = {'section_index': section_index, 'section_chunk_index': section_chunk_index, 'section_char_length': len(section_text), 'char_start': start, 'char_end': end, 'char_length': len(clean_chunk), 'starts_mid_sentence': cls._starts_mid_sentence(section_text, start), 'ends_mid_sentence': cls._ends_mid_sentence(section_text, end), 'content_role': cls._infer_chunk_content_role(section_title, clean_chunk)}
                    new_chunk = {'chunk_index': chunk_index, 'chunk_text': clean_chunk, 'section_title': section_title, 'token_count': cls._rough_token_count(clean_chunk), 'metadata': metadata}
                    if cls._should_absorb_into_previous(chunks, new_chunk):
                        cls._merge_chunk_into_previous(chunks[-1], new_chunk)
                    else:
                        chunks.append(new_chunk)
                        chunk_index += 1
                        section_chunk_index += 1
                if end >= len(section_text):
                    break
                start = cls._adjust_next_chunk_start(section_text, end, overlap_chars)
        cls._refine_chunk_content_roles(chunks)
        cls._annotate_chunk_links(chunks)
        return chunks

    @staticmethod
    def _adjust_chunk_end(text: str, start: int, max_chars: int) -> int:
        tentative_end = min(len(text), start + max_chars)
        if tentative_end >= len(text):
            return len(text)
        window_end = min(len(text), tentative_end + 220)
        candidate_window = text[start:window_end]
        minimum_break = max(max_chars // 2, 220)
        preferred_breaks = [match.end() for match in re.finditer('(?:\\n{2,}|\\.\\s+|\\?\\s+|!\\s+)', candidate_window) if match.end() >= minimum_break]
        if preferred_breaks:
            return start + preferred_breaks[-1]
        line_breaks = [match.end() for match in re.finditer('\\n+', candidate_window) if match.end() >= minimum_break]
        if line_breaks:
            return start + line_breaks[-1]
        last_space = candidate_window.rfind(' ')
        if last_space >= minimum_break:
            return start + last_space + 1
        return tentative_end

    @classmethod
    def _adjust_next_chunk_start(cls, text: str, end: int, overlap_chars: int) -> int:
        start = max(0, end - overlap_chars)
        if start <= 0:
            return 0
        rewind_start = max(0, end - overlap_chars - 160)
        rewind_window = text[rewind_start:end]
        sentence_breaks = list(re.finditer('(?:\\.\\s+|\\?\\s+|!\\s+|\\n{2,})', rewind_window))
        if sentence_breaks:
            start = rewind_start + sentence_breaks[-1].end()
        else:
            sentence_window = text[start:min(len(text), start + 160)]
            sentence_break = re.search('(?:\\.\\s+|\\?\\s+|!\\s+|\\n{2,})', sentence_window)
            if sentence_break is not None:
                start += sentence_break.end()
        boundary_window_end = min(len(text), start + 80)
        while start < boundary_window_end and text[start - 1].isalnum() and text[start].isalnum():
            start += 1
        while start < len(text) and text[start].isspace():
            start += 1
        fragment_window = text[start:min(len(text), start + 260)]
        if cls._looks_like_fragmentary_chunk_start(fragment_window):
            sentence_break = re.search('(?:\\.\\s+|\\?\\s+|!\\s+|\\n{2,})', fragment_window)
            if sentence_break is not None:
                start += sentence_break.end()
                while start < len(text) and text[start].isspace():
                    start += 1
        elif re.match('^[a-z][a-z-]{2,}\\b', fragment_window):
            sentence_break = re.search('(?:\\.\\s+|\\?\\s+|!\\s+|\\n{2,})', fragment_window[:220])
            if sentence_break is not None:
                start += sentence_break.end()
                while start < len(text) and text[start].isspace():
                    start += 1
        return start

    @staticmethod
    def _should_absorb_into_previous(chunks: list[dict[str, Any]], new_chunk: dict[str, Any]) -> bool:
        if not chunks:
            return False
        previous = chunks[-1]
        if previous.get('section_title') != new_chunk.get('section_title'):
            return False
        new_text = str(new_chunk.get('chunk_text') or '')
        if len(new_text) >= 160:
            if not (len(new_text) <= 360 and re.match('^[a-z0-9]', new_text)):
                return False
        metadata = new_chunk.get('metadata') or {}
        if metadata.get('starts_mid_sentence'):
            return True
        return bool(re.fullmatch('[\\d\\s.,;:()[\\]{}%-]{1,360}', new_text)) or len(new_text.split()) <= 12 or bool(re.match('^[a-z0-9]', new_text))

    @classmethod
    def _merge_chunk_into_previous(cls, previous: dict[str, Any], new_chunk: dict[str, Any]) -> None:
        previous_text = str(previous.get('chunk_text') or '').rstrip()
        new_text = str(new_chunk.get('chunk_text') or '').lstrip()
        separator = '\n' if previous_text and (not previous_text.endswith('\n')) else ''
        merged_text = f'{previous_text}{separator}{new_text}'.strip()
        previous['chunk_text'] = merged_text
        previous['token_count'] = cls._rough_token_count(merged_text)
        previous_metadata = previous.setdefault('metadata', {})
        new_metadata = new_chunk.get('metadata') or {}
        previous_metadata['char_end'] = new_metadata.get('char_end', previous_metadata.get('char_end'))
        previous_metadata['char_length'] = len(merged_text)
        previous_metadata['ends_mid_sentence'] = bool(new_metadata.get('ends_mid_sentence'))
        previous_metadata['continues_next'] = bool(new_metadata.get('ends_mid_sentence'))

    @staticmethod
    def _annotate_chunk_links(chunks: list[dict[str, Any]]) -> None:
        for (index, chunk) in enumerate(chunks):
            metadata = chunk.setdefault('metadata', {})
            metadata['prev_chunk_index'] = chunks[index - 1]['chunk_index'] if index > 0 else None
            metadata['next_chunk_index'] = chunks[index + 1]['chunk_index'] if index + 1 < len(chunks) else None
            metadata['continues_previous'] = bool(metadata.get('starts_mid_sentence'))
            metadata['continues_next'] = bool(metadata.get('ends_mid_sentence'))

    @staticmethod
    def _infer_content_role(section_title: str) -> str:
        lowered = section_title.lower()
        if lowered == 'front matter':
            return 'front_matter'
        if is_references_section_title(section_title):
            return 'references'
        if 'appendix' in lowered:
            return 'appendix'
        if 'table of contents' in lowered:
            return 'toc'
        return 'body'

    @classmethod
    def _infer_chunk_content_role(cls, section_title: str, chunk_text: str) -> str:
        section_role = cls._infer_content_role(section_title)
        compact = ' '.join(chunk_text.split())
        if section_role == 'front_matter':
            if cls._looks_like_table_like_chunk(chunk_text, compact):
                return 'table_like'
            return section_role
        if section_role != 'body':
            return section_role
        if cls._looks_like_toc_line(compact):
            return 'toc'
        if cls._looks_like_figure_caption_chunk(compact):
            return 'figure_caption'
        if re.match('^(?:Figure|Table)\\s+\\d+\\s+(?:presents|shows|compares|illustrates|visualizes|reports|summarizes|demonstrates|plots)\\b', compact, re.IGNORECASE):
            return 'body'
        if cls._looks_like_body_chunk(section_title, compact):
            return 'body'
        if cls._looks_like_reference_chunk(compact):
            return 'references'
        if cls._looks_like_table_like_chunk(chunk_text, compact):
            return 'table_like'
        return 'body'

    @classmethod
    def _refine_chunk_content_roles(cls, chunks: list[dict[str, Any]]) -> None:
        for chunk in chunks:
            metadata = chunk.setdefault('metadata', {})
            section_title = str(chunk.get('section_title') or '')
            chunk_text = str(chunk.get('chunk_text') or '')
            compact = ' '.join(chunk_text.split())
            current_role = str(metadata.get('content_role') or 'body')
            section_role = cls._infer_content_role(section_title)
            body_like = cls._looks_like_body_chunk(section_title, compact)
            reference_like = cls._looks_like_reference_chunk(compact)
            if section_role == 'references':
                metadata['content_role'] = 'references'
                continue
            if current_role == 'references' and body_like:
                metadata['content_role'] = 'body'
                continue
            if current_role in {'body', 'appendix'} and reference_like and (not body_like):
                metadata['content_role'] = 'references'
                continue
            if current_role in {'body', 'front_matter'} and (not body_like) and cls._looks_like_table_like_chunk(chunk_text, compact):
                metadata['content_role'] = 'table_like'

    @classmethod
    def _looks_like_body_chunk(cls, section_title: str, compact: str) -> bool:
        if not compact:
            return False
        lowered_title = section_title.lower()
        if any((keyword in lowered_title for keyword in ('introduction', 'method', 'approach', 'experiment', 'result', 'discussion', 'conclusion', 'abstract', 'related work'))):
            if cls._starts_like_body_paragraph(compact[:180], compact):
                return True
        sentence_breaks = compact.count('. ') + compact.count('? ') + compact.count('! ')
        alphabetic_tokens = re.findall('[A-Za-z]{3,}', compact[:260])
        citation_markers = len(re.findall('\\[\\d+\\]', compact[:260]))
        if sentence_breaks >= 1 and len(alphabetic_tokens) >= 12 and (citation_markers <= 8):
            if re.match('^[A-Z][a-z]', compact):
                return True
        return False

    @classmethod
    def _looks_like_reference_chunk(cls, compact: str) -> bool:
        compact_prefix = compact[:260]
        if not compact_prefix:
            return False
        if cls._looks_like_body_chunk('Full Text', compact_prefix):
            return False
        if re.match('^(?:\\[\\d+\\]|\\d+\\.)\\s+[A-Z]', compact_prefix):
            return True
        if re.match('^\\d+\\s+\\[\\d+\\]\\s+[A-Z]', compact_prefix):
            return True
        if re.match('^(?:\\d+\\s*,?\\s*)?\\[\\d+\\]\\s+[A-Z]', compact_prefix):
            return True
        if compact_prefix.startswith('In ') and re.search('\\b(?:Conference|Proceedings|arXiv|Association for Computational Linguistics)\\b', compact_prefix):
            return True
        if re.match('^\\d', compact_prefix) and re.search('\\[\\d+\\]', compact_prefix) and re.search('\\b(?:19|20)\\d{2}\\b', compact_prefix):
            return True
        if compact_prefix.startswith(('doi:', 'https://', 'http://')) and re.search('\\[\\d+\\]', compact_prefix):
            return True
        if re.match("^[A-Z][A-Za-z'`.-]+(?:,\\s+[A-Z][A-Za-z'`.-]+){1,}", compact_prefix) and re.search('\\b(?:19|20)\\d{2}\\b', compact_prefix):
            return True
        return len(re.findall('(?:^|\\s)\\d+\\s+\\[\\d+\\]', compact[:220])) >= 1

    @staticmethod
    def _looks_like_figure_caption_chunk(compact: str) -> bool:
        if not compact:
            return False
        explicit_caption = re.match('^(?:Figure|Table)\\s+\\d+[:.]', compact)
        panel_caption = re.match('^\\d+\\.\\s*[A-Z][^.]{0,140}\\([a-z]\\)', compact)
        multi_panel_caption = compact.startswith(('a, ', 'b, ', 'c, ', 'd, ')) and re.search('\\b[bcd],\\s+[A-Z]', compact[:220])
        if not (explicit_caption or panel_caption or multi_panel_caption):
            return False
        if re.match('^(?:Figure|Table)\\s+\\d+\\s+(?:presents|shows|compares|illustrates|visualizes|reports|summarizes|demonstrates|plots)\\b', compact, re.IGNORECASE):
            return False
        return True

    @classmethod
    def _looks_like_table_like_chunk(cls, raw_text: str, compact: str) -> bool:
        if '/uni' in raw_text:
            return True
        if any((token in raw_text.lower() for token in ('<td', '</td', '<tr', '</tr', '<th', '</th'))):
            return True
        digits = sum((character.isdigit() for character in compact))
        numeric_cells = len(re.findall('\\b\\d+(?:\\.\\d+)?\\b', compact))
        symbol_cells = len(re.findall('[✓✔✗✘]', compact))
        percentage_cells = len(re.findall('\\b\\d+(?:\\.\\d+)?%', compact))
        short_code_cells = len(re.findall('\\b(?:MCQ|OE|A&M|IoU|FVD|PSNR|SSIM|M|A|S|T|R|F1|Top-1|Top-5)\\b', compact, re.IGNORECASE))
        line_break_count = raw_text.count('\n')
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        first_line = lines[0] if lines else compact[:160]
        numeric_heavy_lines = sum((1 for line in lines if len(re.findall('\\b\\d+(?:\\.\\d+)?\\b', line)) >= 2 or bool(re.fullmatch('[\\d\\s.,:;()%+\\-=/]+', line))))
        repeated_matrix_tokens = len(re.findall('\\b[a-z]+-\\d+-(?:combined|pre|post)\\b', compact, re.IGNORECASE))
        first_line_numeric_cells = len(re.findall('\\b\\d+(?:\\.\\d+)?\\b', first_line))
        explicit_tabular_opening = bool(re.match('^[\\d.,;:()%-]', compact) or re.match('^(?:Table|Figure)\\s+\\d+[:.]', compact) or re.match('^\\d+\\.\\s*[A-Z][^.]{0,140}\\([a-z]\\)', compact) or (repeated_matrix_tokens >= 6))
        compact_row_like_block = numeric_cells >= 10 and (symbol_cells >= 2 or percentage_cells >= 2 or short_code_cells >= 3) and (len(re.findall('\\b[A-Z][A-Za-z-]{2,}\\b', compact)) >= 4)
        strong_numeric_block = compact and digits / len(compact) > 0.28 and (line_break_count >= 5) or (re.match('^[\\d.,;:()%-]', compact) and numeric_cells >= 10 and (line_break_count >= 2)) or (re.match('^(?:Table|Figure)\\s+\\d+\\b', compact) and numeric_heavy_lines >= 3 and (line_break_count >= 2)) or (repeated_matrix_tokens >= 6) or (len(lines) >= 8 and numeric_heavy_lines >= max(6, int(len(lines) * 0.7)) and (digits / max(1, len(compact)) > 0.08)) or compact_row_like_block
        if cls._starts_like_body_paragraph(first_line, compact) and (not explicit_tabular_opening):
            return False
        if compact and digits / len(compact) > 0.28 and (line_break_count >= 5):
            return True
        if re.match('^[\\d.,;:()%-]', compact) and numeric_cells >= 10 and (line_break_count >= 2):
            return True
        if re.search('\\b(?:Table|Figure)\\s+\\d+\\b', compact[:160]) and numeric_heavy_lines >= 3 and (line_break_count >= 2):
            return True
        if repeated_matrix_tokens >= 6:
            return True
        if len(lines) >= 8 and numeric_heavy_lines >= max(6, int(len(lines) * 0.7)) and (digits / max(1, len(compact)) > 0.08):
            return True
        if compact_row_like_block:
            return True
        return False

    @staticmethod
    def _looks_like_fragmentary_chunk_start(text: str) -> bool:
        compact = text.lstrip()
        return bool(re.match('^\\d{4}[a-z]?\\](?:,)?\\s+[a-z]', compact) or re.match('^\\(\\d+\\)\\s+[a-z]', compact) or re.match('^\\d+[a-z]?\\),\\s+[a-z]', compact) or re.match('^\\d+[a-z]\\),', compact) or re.match('^\\d+[a-z]?,\\s+[a-z]', compact) or re.match('^\\d+[A-Z][a-z]', compact))

    @staticmethod
    def _rough_token_count(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def summarize_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
        if not chunks:
            return {'chunk_count': 0, 'body_chunk_count': 0, 'avg_chunk_chars': 0, 'max_chunk_chars': 0, 'avg_chunk_tokens': 0, 'max_chunk_tokens': 0, 'suspicious_start_count': 0, 'body_suspicious_start_count': 0, 'mid_sentence_start_count': 0, 'mid_sentence_end_count': 0, 'front_matter_chunk_count': 0, 'non_body_chunk_count': 0, 'reference_chunk_count': 0, 'table_like_chunk_count': 0, 'tiny_chunk_count': 0}
        char_lengths = [len(str(chunk.get('chunk_text') or '')) for chunk in chunks]
        token_counts = [int(chunk.get('token_count', 0) or 0) for chunk in chunks]
        suspicious_start_count = 0
        body_suspicious_start_count = 0
        for chunk in chunks:
            text = str(chunk.get('chunk_text') or '')
            if text and re.match('^[a-z0-9,.;:)\\]%-]', text):
                suspicious_start_count += 1
                if (chunk.get('metadata') or {}).get('content_role') == 'body':
                    body_suspicious_start_count += 1
        return {'chunk_count': len(chunks), 'body_chunk_count': sum((1 for chunk in chunks if (chunk.get('metadata') or {}).get('content_role') == 'body')), 'avg_chunk_chars': round(sum(char_lengths) / len(char_lengths), 2), 'max_chunk_chars': max(char_lengths), 'avg_chunk_tokens': round(sum(token_counts) / len(token_counts), 2), 'max_chunk_tokens': max(token_counts), 'suspicious_start_count': suspicious_start_count, 'body_suspicious_start_count': body_suspicious_start_count, 'mid_sentence_start_count': sum((1 for chunk in chunks if bool((chunk.get('metadata') or {}).get('starts_mid_sentence')))), 'mid_sentence_end_count': sum((1 for chunk in chunks if bool((chunk.get('metadata') or {}).get('ends_mid_sentence')))), 'front_matter_chunk_count': sum((1 for chunk in chunks if chunk.get('section_title') == 'Front Matter')), 'non_body_chunk_count': sum((1 for chunk in chunks if (chunk.get('metadata') or {}).get('content_role') != 'body')), 'reference_chunk_count': sum((1 for chunk in chunks if (chunk.get('metadata') or {}).get('content_role') == 'references')), 'table_like_chunk_count': sum((1 for chunk in chunks if (chunk.get('metadata') or {}).get('content_role') == 'table_like')), 'tiny_chunk_count': sum((1 for chunk in chunks if len(str(chunk.get('chunk_text') or '')) < 160))}

    @staticmethod
    def _starts_mid_sentence(text: str, start: int) -> bool:
        if start <= 0:
            return False
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            return False
        previous_index = start - 1
        while previous_index >= 0 and text[previous_index].isspace():
            previous_index -= 1
        if previous_index < 0:
            return False
        previous_char = text[previous_index]
        current_char = text[start]
        if previous_char in '.?!:\n':
            return False
        if current_char in ',.;:)]}%-':
            return True
        if previous_char.isalnum() and current_char.isalnum():
            return True
        return current_char.islower() or current_char.isdigit()

    @staticmethod
    def _ends_mid_sentence(text: str, end: int) -> bool:
        if end >= len(text):
            return False
        current_index = max(0, end - 1)
        while current_index >= 0 and text[current_index].isspace():
            current_index -= 1
        if current_index < 0:
            return False
        next_index = end
        while next_index < len(text) and text[next_index].isspace():
            next_index += 1
        if next_index >= len(text):
            return False
        current_char = text[current_index]
        next_char = text[next_index]
        if current_char in '.?!:\n':
            return False
        if next_char in ',.;:)]}%-':
            return True
        if current_char.isalnum() and next_char.isalnum():
            return True
        return next_char.islower() or next_char.isdigit()

