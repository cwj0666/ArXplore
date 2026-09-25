import os

from django.core.asgi import get_asgi_application

from arxplore_web.bootstrap import configure_environment

configure_environment()
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "arxplore_web.settings")

application = get_asgi_application()
