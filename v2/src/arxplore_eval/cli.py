import typer

from arxplore_eval.commands import candidates, fusion, generation, latency, queries, retrieval

app = typer.Typer(no_args_is_help=True, add_completion=False, help="ArXplore evaluation harness")

for module in (retrieval, generation, fusion, latency, queries, candidates):
    module.register(app)


if __name__ == "__main__":
    app()
