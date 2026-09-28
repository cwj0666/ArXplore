import copy
import json
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
from uvicorn.config import LOGGING_CONFIG

APP_FACTORY = "arxplore.api.app:create_app"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
DEFAULT_WORKERS = 1
APP_LOGGER = "arxplore"


def log_config() -> dict[str, Any]:
    config = copy.deepcopy(LOGGING_CONFIG)
    config["loggers"][APP_LOGGER] = {"handlers": ["default"], "level": "INFO", "propagate": False}
    return config


def serve(*, host: str, port: int, workers: int, reload: bool) -> None:
    uvicorn.run(
        APP_FACTORY,
        factory=True,
        host=host,
        port=port,
        workers=None if reload else workers,
        reload=reload,
        proxy_headers=False,
        log_config=log_config(),
    )


def render_openapi() -> str:
    from arxplore.api.app import create_app

    schema = create_app().openapi()
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def register(app: typer.Typer) -> None:
    @app.command("api", help="Serve the HTTP API with uvicorn (FastAPI app factory arxplore.api.app:create_app).")
    def api_command(
        host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = DEFAULT_HOST,
        port: Annotated[int, typer.Option("--port", help="Port to bind.")] = DEFAULT_PORT,
        workers: Annotated[int, typer.Option("--workers", min=1, help="Worker processes.")] = DEFAULT_WORKERS,
        reload: Annotated[bool, typer.Option("--reload", help="Restart on source changes (development).")] = False,
    ) -> None:
        serve(host=host, port=port, workers=workers, reload=reload)

    @app.command("openapi", help="Write the API OpenAPI schema as JSON (sorted keys) for generated client types.")
    def openapi_command(
        output: Annotated[
            Path | None, typer.Option("--output", "-o", help="File to write; standard output when omitted.")
        ] = None,
    ) -> None:
        document = render_openapi()
        if output is None:
            typer.echo(document, nl=False)
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(document, encoding="utf-8")
        typer.echo(f"wrote {output}")
