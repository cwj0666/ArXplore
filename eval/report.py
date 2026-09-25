"""평가 결과를 CSV와 Markdown으로 기록한다."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from eval.runner import BASE_METHODS, METHODS, AggregateRow, QueryResult, metric_names

MISSING = "n/a"
README_TABLE_HEADER = "| 검색 방식 | hit@1 | hit@5 | hit@10 | MRR@10 | 지연 p50 / p95 (ms) |"


def format_rate(value: float | None) -> str:
    return MISSING if value is None else f"{value:.3f}"


def format_ms(value: float | None) -> str:
    return MISSING if value is None else f"{value:.0f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _metric_label(name: str) -> str:
    return name.replace("mrr@", "MRR@")


def render_readme_table(aggregates: Sequence[AggregateRow]) -> str:
    """README Evaluation 표와 같은 모양(전체 언어, 기본 3방식)."""
    by_method = {row.method: row for row in aggregates if row.subset == "all"}
    lines = [README_TABLE_HEADER, "| --- | --- | --- | --- | --- | --- |"]
    for method in BASE_METHODS:
        row = by_method.get(method)
        if row is None:
            continue
        lines.append(
            "| {method} | {h1} | {h5} | {h10} | {mrr} | {p50} / {p95} |".format(
                method=method,
                h1=format_rate(row.paper.get("hit@1")),
                h5=format_rate(row.paper.get("hit@5")),
                h10=format_rate(row.paper.get("hit@10")),
                mrr=format_rate(row.paper.get(f"mrr@{row.k}")),
                p50=format_ms(row.latency_p50_ms),
                p95=format_ms(row.latency_p95_ms),
            )
        )
    return "\n".join(lines)


def render_markdown(aggregates: Sequence[AggregateRow], *, title: str, meta: Mapping[str, str]) -> str:
    lines = [f"# {title}", ""]
    lines.extend(f"- {key}: {value}" for key, value in meta.items())
    lines.append("")

    if not aggregates:
        lines.append("집계할 결과가 없습니다.")
        return "\n".join(lines) + "\n"

    k = aggregates[0].k
    names = metric_names(k)
    labels = [_metric_label(name) for name in names]
    overall = [row for row in aggregates if row.subset == "all"]

    lines += ["## README 붙여넣기용 (논문 단위, 전체 언어)", "", render_readme_table(aggregates), ""]

    lines += ["## 논문 단위 (arxiv_id 기준)", ""]
    lines += _table(
        ["method", "subset", "n", *labels, f"noise@{k}", "p50 ms", "p95 ms", "errors"],
        [
            [
                row.method,
                row.subset,
                str(row.n_queries),
                *(format_rate(row.paper.get(name)) for name in names),
                format_rate(row.noise_rate),
                format_ms(row.latency_p50_ms),
                format_ms(row.latency_p95_ms),
                str(row.n_errors),
            ]
            for row in aggregates
        ],
    )
    lines.append("")

    chunk_rows = [row for row in aggregates if row.n_chunk_queries]
    lines += ["## 청크 단위 (relevant_chunk_ids가 있는 질의만)", ""]
    if chunk_rows:
        lines += _table(
            ["method", "subset", "n", *labels],
            [
                [
                    row.method,
                    row.subset,
                    str(row.n_chunk_queries),
                    *(format_rate(row.chunk.get(name)) for name in names),
                ]
                for row in chunk_rows
            ],
        )
    else:
        lines.append("청크 단위 정답이 있는 질의가 없습니다.")
    lines.append("")

    lines += ["## 방식", ""]
    lines.extend(f"- `{row.method}`: {METHODS[row.method].description}" for row in overall if row.method in METHODS)
    lines += [
        "",
        f"noise@{k}: 상위 {k}개 hit 중 content_role이 references/toc/front_matter인 비율. "
        "지연은 문맥 창 조회까지 포함한 방식 호출 1회의 wall-clock.",
    ]
    return "\n".join(lines) + "\n"


QUERY_CSV_BASE_FIELDS = [
    "query_id",
    "lang",
    "source",
    "category",
    "expected_behavior",
    "method",
    "k",
    "latency_ms",
    "error",
]


def write_query_csv(results: Sequence[QueryResult], path: str | Path) -> None:
    k = results[0].k if results else 10
    names = metric_names(k)
    fields = [
        *QUERY_CSV_BASE_FIELDS,
        *(f"paper_{name}" for name in names),
        *(f"chunk_{name}" for name in names),
        "noise_count",
        "retrieved_arxiv_ids",
        "retrieved_chunk_ids",
        "retrieved_roles",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            record: dict[str, object] = {
                "query_id": row.query_id,
                "lang": row.lang,
                "source": row.source,
                "category": row.category,
                "expected_behavior": row.expected_behavior,
                "method": row.method,
                "k": row.k,
                "latency_ms": "" if row.latency_ms is None else f"{row.latency_ms:.1f}",
                "error": row.error or "",
                "noise_count": row.noise_count,
                "retrieved_arxiv_ids": " ".join(row.retrieved_arxiv_ids),
                "retrieved_chunk_ids": " ".join(str(value) for value in row.retrieved_chunk_ids),
                "retrieved_roles": " ".join(role or "-" for role in row.retrieved_roles),
            }
            for name in names:
                paper_value = row.paper_metrics.get(name)
                chunk_value = row.chunk_metrics.get(name)
                record[f"paper_{name}"] = "" if paper_value is None else f"{paper_value:.4f}"
                record[f"chunk_{name}"] = "" if chunk_value is None else f"{chunk_value:.4f}"
            writer.writerow(record)


def write_summary_csv(aggregates: Sequence[AggregateRow], path: str | Path) -> None:
    k = aggregates[0].k if aggregates else 10
    names = metric_names(k)
    fields = [
        "method",
        "subset",
        "n_queries",
        "n_errors",
        *(f"paper_{name}" for name in names),
        "n_chunk_queries",
        *(f"chunk_{name}" for name in names),
        "noise_rate",
        "latency_p50_ms",
        "latency_p95_ms",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in aggregates:
            record: dict[str, object] = {
                "method": row.method,
                "subset": row.subset,
                "n_queries": row.n_queries,
                "n_errors": row.n_errors,
                "n_chunk_queries": row.n_chunk_queries,
                "noise_rate": "" if row.noise_rate is None else f"{row.noise_rate:.4f}",
                "latency_p50_ms": "" if row.latency_p50_ms is None else f"{row.latency_p50_ms:.1f}",
                "latency_p95_ms": "" if row.latency_p95_ms is None else f"{row.latency_p95_ms:.1f}",
            }
            for name in names:
                paper_value = row.paper.get(name)
                chunk_value = row.chunk.get(name)
                record[f"paper_{name}"] = "" if paper_value is None else f"{paper_value:.4f}"
                record[f"chunk_{name}"] = "" if chunk_value is None else f"{chunk_value:.4f}"
            writer.writerow(record)
