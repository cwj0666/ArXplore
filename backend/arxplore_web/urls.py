from django.conf import settings
from django.contrib import admin
from django.urls import include, path

from papers import api_views, page_views


def admin_urlpatterns() -> list:
    if not settings.ADMIN_ENABLED:
        return []
    return [path(settings.ADMIN_PATH, admin.site.urls)]


urlpatterns = [
    *admin_urlpatterns(),
    path("bootstrap.json", api_views.bootstrap, name="bootstrap"),
    path("auth/signup/", api_views.auth_signup, name="auth_signup"),
    path("auth/login/", api_views.auth_login, name="auth_login"),
    path("auth/logout/", api_views.auth_logout, name="auth_logout"),
    path("auth/me/", api_views.auth_me, name="auth_me"),
    path("settings/", api_views.settings_detail, name="settings_detail"),
    path("settings/api-key/", api_views.settings_api_key_detail, name="settings_api_key_detail"),
    path("favorites/", api_views.favorites_list, name="favorites_list"),
    path("favorites/toggle/", api_views.favorites_toggle, name="favorites_toggle"),
    path("papers/", include("papers.urls")),
    path("login/", page_views.login_page, name="login_page"),
    path("", page_views.paper_list),
]
