from django.urls import path
from .views import PredictView
from .genai import Detect, VideoStreamDetectionView, VideoStreamMJPEGView

urlpatterns = [
    path('predict/', PredictView.as_view(), name='predict'),
    path('detect_damage/', Detect.as_view(), name='detect_damage'),
    #path("detect/", LiveYOLODetectionView.as_view(), name="live-detection"),
    path("detect/", VideoStreamDetectionView.as_view(), name="detect"),
    path('live-stream/', VideoStreamMJPEGView.as_view(), name='live_stream'),
]
