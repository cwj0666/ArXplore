import typer

from arxplore_eval.commands import (
    candidates,
    compare,
    compare_generation,
    cost,
    explain,
    fusion,
    generation,
    latency,
    plan_diff,
    queries,
    remap,
    retrieval,
)

app = typer.Typer(no_args_is_help=True, add_completion=False, help="ArXplore evaluation harness")

for module in (
    retrieval,
    generation,
    fusion,
    latency,
    queries,
    candidates,
    explain,
    remap,
    compare,
    compare_generation,
    cost,
    plan_diff,
):
    module.register(app)


if __name__ == "__main__":
    app()
