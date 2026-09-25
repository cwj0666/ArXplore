from django.conf import settings
from django.db import models

DEFAULT_SUMMARY_MODEL = "gpt-5-mini"


class UserSettings(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="paper_settings")
    preferred_summary_model = models.CharField(max_length=100, default=DEFAULT_SUMMARY_MODEL)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "user_settings"


class FavoritePaper(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="favorite_papers")
    arxiv_id = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "favorite_papers"
        constraints = [
            models.UniqueConstraint(fields=["user", "arxiv_id"], name="favorite_papers_user_arxiv_unique"),
        ]
