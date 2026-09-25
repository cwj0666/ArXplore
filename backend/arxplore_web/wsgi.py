import os

from django.core.wsgi import get_wsgi_application

from arxplore_web.bootstrap import configure_environment

configure_environment()
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "arxplore_web.settings")

application = get_wsgi_application()
