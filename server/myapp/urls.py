from django.urls import path
from .views import PredictView
from .genai import Detect

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('detect_damage/', Detect.as_view(), name='detect_damage')
]
