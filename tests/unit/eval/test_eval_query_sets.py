from arxplore_eval.build_queries import CASE_QUERIES_FILE, COMBINED_QUERIES_FILE, GENERATED_QUERIES_FILE
from arxplore_eval.dataset import find_placeholders, load_queries, placeholder_problems


def test_queries_jsonl_is_generated_then_cases_byte_for_byte(eval_dir) -> None:
    generated = (eval_dir / GENERATED_QUERIES_FILE).read_bytes()
    cases = (eval_dir / CASE_QUERIES_FILE).read_bytes()
    combined = (eval_dir / COMBINED_QUERIES_FILE).read_bytes()

    assert generated.endswith(b"\n") and cases.endswith(b"\n")
    assert combined == generated + cases


def test_combined_records_keep_the_source_order_and_unique_ids(eval_dir) -> None:
    generated = load_queries(eval_dir / GENERATED_QUERIES_FILE)
    cases = load_queries(eval_dir / CASE_QUERIES_FILE)
    combined = load_queries(eval_dir / COMBINED_QUERIES_FILE)

    assert combined == generated + cases
    assert len({query.id for query in combined}) == len(combined)


def test_combined_query_set_has_no_placeholders(eval_dir) -> None:
    combined = load_queries(eval_dir / COMBINED_QUERIES_FILE)

    assert placeholder_problems(combined) == []
    assert find_placeholders([query.to_dict() for query in combined]) == set()
    assert any(query.retrieval_eligible for query in combined)
