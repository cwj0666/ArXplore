import typer

from arxplore_eval.commands import candidates, explain, fusion, generation, latency, queries, remap, retrieval

app = typer.Typer(no_args_is_help=True, add_completion=False, help="ArXplore evaluation harness")

for module in (retrieval, generation, fusion, latency, queries, candidates, explain, remap):
    module.register(app)


if __name__ == "__main__":
    app()
