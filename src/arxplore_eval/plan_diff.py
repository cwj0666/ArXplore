import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from arxplore.domain.retrieval import RetrievalHit
from arxplore_eval.agent_plan import AgentPlan
from arxplore_eval.dataset import EvalQuery
from arxplore_eval.fusion_variants import FUSION_CONVEX, FUSION_LEGACY_RULES, fuse_inputs
from arxplore_eval.tables import markdown_table

SAME = "same"
REORDERED = "reordered"
CHANGED = "changed"
NO_SEARCH = "no_search"
STRATA = (SAME, REORDERED, CHANGED, NO_SEARCH)
RELATION_RANK = {SAME: 0, REORDERED: 1, CHANGED: 2}
STRATUM_DESCRIPTIONS = {
    SAME: "모든 검색 호출에서 두 융합의 top-k chunk 순서가 같음",
    REORDERED: "top-k chunk 집합은 같고 순서만 다른 호출이 있음",
    CHANGED: "top-k chunk 집합이 다른 호출이 있음",
    NO_SEARCH: "첫 턴에 검색 호출이 없음(두 조건의 입력이 같음)",
}
MD_FILE = "plan_diff.md"
SEARCHES_CSV_FILE = "plan_diff_searches.csv"
STRATA_CSV_FILE = "strata.csv"


@dataclass(frozen=True)
class SearchDiff:
    query_id: str
    tool_call_id: str
    tool_query: str
    a_chunks: tuple[int, ...]
    b_chunks: tuple[int, ...]
    a_papers: tuple[str, ...]
    b_papers: tuple[str, ...]
    a_gold_rank: int | None
    b_gold_rank: int | None

    @property
    def relation(self) -> str:
        if self.a_chunks == self.b_chunks:
            return SAME
        if set(self.a_chunks) == set(self.b_chunks):
            return REORDERED
        return CHANGED

    @property
    def jaccard(self) -> float:
        union = set(self.a_chunks) | set(self.b_chunks)
        return len(set(self.a_chunks) & set(self.b_chunks)) / len(union) if union else 1.0


@dataclass(frozen=True)
class QueryDiff:
    query: EvalQuery
    kind: str
    searches: tuple[SearchDiff, ...]

    @property
    def stratum(self) -> str:
        if not self.searches:
            return NO_SEARCH
        return max((search.relation for search in self.searches), key=RELATION_RANK.__getitem__)

    def gold_hit(self, side: str) -> bool:
        return any(getattr(search, f"{side}_gold_rank") is not None for search in self.searches)


def gold_rank(hits: Sequence[RetrievalHit], relevant: set[str]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if hit.arxiv_id in relevant:
            return rank
    return None


def diff_plan(plan: AgentPlan, query: EvalQuery) -> QueryDiff:
    relevant = set(query.relevant_arxiv_ids)
    searches: list[SearchDiff] = []
    for search in plan.searches:
        a = fuse_inputs(search.inputs, FUSION_LEGACY_RULES, limit=plan.search_limit)
        b = fuse_inputs(search.inputs, FUSION_CONVEX, limit=plan.search_limit)
        searches.append(
            SearchDiff(
                query_id=plan.query_id,
                tool_call_id=search.tool_call_id,
                tool_query=search.query,
                a_chunks=tuple(hit.chunk_id for hit in a),
                b_chunks=tuple(hit.chunk_id for hit in b),
                a_papers=tuple(dict.fromkeys(hit.arxiv_id for hit in a)),
                b_papers=tuple(dict.fromkeys(hit.arxiv_id for hit in b)),
                a_gold_rank=gold_rank(a, relevant),
                b_gold_rank=gold_rank(b, relevant),
            )
        )
    return QueryDiff(query=query, kind=plan.kind, searches=tuple(searches))


def diff_plans(plans: Mapping[str, AgentPlan], queries: Sequence[EvalQuery]) -> list[QueryDiff]:
    return [diff_plan(plans[query.id], query) for query in queries if query.id in plans]


def _count_row(label: str, diffs: Sequence[QueryDiff]) -> list[str]:
    counts = Counter(diff.stratum for diff in diffs)
    return [label, str(len(diffs)), *(str(counts.get(stratum, 0)) for stratum in STRATA)]


def _rank_text(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def render_markdown(diffs: Sequence[QueryDiff], *, title: str, meta: Mapping[str, str]) -> str:
    lines = [f"# {title}", ""]
    lines.extend(f"- {key}: {value}" for key, value in meta.items())
    lines += ["", "A = legacy_rules(옛 가중 RRF), B = convex(제품 min-max 가중합). 같은 융합 전 후보에서 계산했다.", ""]
    lines += ["## 질의 층", ""]
    groups: list[tuple[str, list[QueryDiff]]] = [("all", list(diffs))]
    for lang in ("ko", "en"):
        groups.append((lang, [diff for diff in diffs if diff.query.lang == lang]))
    for source in sorted({diff.query.source for diff in diffs}):
        groups.append((source, [diff for diff in diffs if diff.query.source == source]))
    lines += markdown_table(["subset", "질의", *STRATA], [_count_row(label, group) for label, group in groups if group])
    lines.append("")
    lines.extend(f"- `{stratum}`: {STRATUM_DESCRIPTIONS[stratum]}" for stratum in STRATA)
    searches = [search for diff in diffs for search in diff.searches]
    lines += ["", "## 검색 호출", ""]
    if searches:
        relations = Counter(search.relation for search in searches)
        mean_jaccard = sum(search.jaccard for search in searches) / len(searches)
        lines += [
            f"- 검색 호출 {len(searches)}개: "
            + ", ".join(f"{name} {relations.get(name, 0)}" for name in (SAME, REORDERED, CHANGED)),
            f"- top-k chunk Jaccard 평균 {mean_jaccard:.3f}",
        ]
    else:
        lines.append("- 검색 호출이 없습니다.")
    hit_a = sum(diff.gold_hit("a") for diff in diffs if diff.searches)
    hit_b = sum(diff.gold_hit("b") for diff in diffs if diff.searches)
    with_search = sum(1 for diff in diffs if diff.searches)
    lines += [
        f"- 정답 논문이 어느 검색 호출의 top-k에든 들어간 질의: A {hit_a}/{with_search}, B {hit_b}/{with_search}",
        "",
        "## 질의별",
        "",
    ]
    rows = [
        [
            diff.query.id,
            diff.query.lang,
            diff.stratum,
            str(len(diff.searches)),
            " / ".join(f"{search.jaccard:.2f}" for search in diff.searches) or "-",
            " / ".join(_rank_text(search.a_gold_rank) for search in diff.searches) or "-",
            " / ".join(_rank_text(search.b_gold_rank) for search in diff.searches) or "-",
        ]
        for diff in diffs
    ]
    lines += markdown_table(["id", "lang", "층", "검색 수", "Jaccard", "정답 순위 A", "정답 순위 B"], rows)
    return "\n".join(lines) + "\n"


def _ids(values: Sequence[object]) -> str:
    return " ".join(str(value) for value in values)


def write_searches_csv(diffs: Sequence[QueryDiff], path: Path) -> None:
    fields = [
        "id",
        "lang",
        "source",
        "stratum",
        "tool_call_id",
        "tool_query",
        "relation",
        "jaccard",
        "a_chunk_ids",
        "b_chunk_ids",
        "a_arxiv_ids",
        "b_arxiv_ids",
        "a_gold_rank",
        "b_gold_rank",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for diff in diffs:
            for search in diff.searches:
                writer.writerow(
                    {
                        "id": diff.query.id,
                        "lang": diff.query.lang,
                        "source": diff.query.source,
                        "stratum": diff.stratum,
                        "tool_call_id": search.tool_call_id,
                        "tool_query": search.tool_query,
                        "relation": search.relation,
                        "jaccard": f"{search.jaccard:.4f}",
                        "a_chunk_ids": _ids(search.a_chunks),
                        "b_chunk_ids": _ids(search.b_chunks),
                        "a_arxiv_ids": _ids(search.a_papers),
                        "b_arxiv_ids": _ids(search.b_papers),
                        "a_gold_rank": "" if search.a_gold_rank is None else search.a_gold_rank,
                        "b_gold_rank": "" if search.b_gold_rank is None else search.b_gold_rank,
                    }
                )


def write_strata_csv(diffs: Sequence[QueryDiff], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "stratum"])
        writer.writeheader()
        for diff in diffs:
            writer.writerow({"id": diff.query.id, "stratum": diff.stratum})


def read_strata_csv(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"id", "stratum"} <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: id, stratum 열이 필요합니다")
        return {row["id"]: row["stratum"] for row in reader if row["id"] and row["stratum"]}
