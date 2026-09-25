from django.urls import include, path

from papers import api_views, page_views

urlpatterns = [
    path("bootstrap.json", api_views.bootstrap, name="bootstrap"),
    path("papers/", include("papers.urls")),
    path("", page_views.paper_list),
]
