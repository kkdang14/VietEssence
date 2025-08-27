from django.conf.urls.static import static
from django.conf import settings
from django.contrib import admin
from django.urls import path
from .views import ImageClassificationAPI, ClassificationResultsAPI

urlpatterns = [
    path('api/classify/', ImageClassificationAPI.as_view(), name='image_classification'),
    path('api/get_storage', ClassificationResultsAPI.as_view(), name='get_storage'),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)