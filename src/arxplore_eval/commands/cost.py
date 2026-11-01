import json
from pathlib import Path
from typing import Annotated

import typer
from psycopg import Error as DatabaseError

from arxplore_eval.commands.common import echo_err, fail
from arxplore_eval.embed_cost import measure_cost
from arxplore_eval.embedders import token_counter


def register(app: typer.Typer) -> None:
    @app.command(
        "embed-cost",
        help="임베딩 대상 청크의 토큰 수와 비용 추정을 낸다(읽기 전용 트랜잭션, API 호출 없음).",
    )
    def embed_cost_command(
        db: Annotated[str, typer.Option("--db", help="PostgreSQL DSN (예: postgresql://user@host:port/db)")],
        model: Annotated[str, typer.Option(help="추정 대상 임베딩 모델 이름(표시용)")],
        price_per_million: Annotated[float, typer.Option(min=0.0, help="1M 토큰당 USD")],
        all_chunks: Annotated[
            bool,
            typer.Option(
                "--all/--missing",
                help="--all: references를 뺀 전체 청크. --missing(기본): worker가 고를 임베딩 없는 청크",
            ),
        ] = False,
        json_out: Annotated[Path | None, typer.Option(help="결과를 JSON으로도 쓴다")] = None,
    ) -> None:
        counter, token_method = token_counter()
        try:
            report = measure_cost(db, all_chunks=all_chunks, counter=counter, token_method=token_method)
        except DatabaseError as exc:
            fail(f"PostgreSQL에 연결할 수 없습니다: {type(exc).__name__}: {exc}")
        summary = report.summary
        scope = "references 제외 전체 청크" if all_chunks else "임베딩 없는 청크(references 제외, worker와 같은 조건)"
        lines = [
            f"대상: {scope}",
            f"모델: {model}",
            f"청크: {summary.chunks}개 (빈 텍스트 {summary.empty_chunks}개)",
            f"글자: {summary.characters}",
            f"토큰: {summary.tokens} ({summary.token_method})",
            f"추정 비용: ${summary.usd(price_per_million):.4f} (${price_per_million}/1M 토큰)",
            "현재 paper_embeddings: "
            + (", ".join(f"{name} {count}" for name, count in report.embeddings.items()) or "비어 있음"),
        ]
        for line in lines:
            typer.echo(line)
        if json_out is not None:
            payload = {
                "scope": "all" if all_chunks else "missing",
                "model": model,
                **summary.as_dict(price_per_million),
                "paper_embeddings": report.embeddings,
            }
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            echo_err(f"결과: {json_out}")
