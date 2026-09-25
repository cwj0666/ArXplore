from django.urls import path

from . import api_views, page_views

urlpatterns = [
    path('', page_views.paper_list_alias, name='paper_list'),
    path('assistant/', page_views.paper_agent, name='paper_agent'),
    path('assistant/chat/', api_views.paper_agent_chat, name='paper_agent_chat'),
    path('assistant/stream/', api_views.paper_agent_stream, name='paper_agent_stream'),
    path('list.json', page_views.paper_list_data, name='paper_list_data'),
    path('<str:arxiv_id>/detail.json', page_views.paper_detail_data, name='paper_detail_data'),
    path('<str:arxiv_id>/analyze/', api_views.paper_analyze, name='paper_analyze'),
    path('<str:arxiv_id>/summary/', api_views.paper_summary, name='paper_summary'),
    path('<str:arxiv_id>/chat/', api_views.paper_chat, name='paper_chat'),
    path('<str:arxiv_id>/chat/stream/', api_views.paper_chat_stream, name='paper_chat_stream'),
    path('<str:arxiv_id>/', page_views.paper_detail, name='paper_detail'),
]
