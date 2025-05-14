from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import uuid
import requests
import cv2
from dotenv import load_dotenv
from ultralytics import YOLO
import base64
from .generative import generate_road_damage_report, generate_stream
from django.http import StreamingHttpResponse


load_dotenv()
model_path = os.getenv("MODEL_PATH2")
model = YOLO(model_path)

class Detect(APIView):
    def post(self, request, *args, **kwargs):
        image_file = request.FILES.get('image')

        if not image_file:
            return Response({'error': 'Image file not provided.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            file_ext = os.path.splitext(image_file.name)[-1]
            file_name = f"{uuid.uuid4()}{file_ext}"
            file_path = os.path.join('uploads', file_name)

            saved_path = default_storage.save(file_path, ContentFile(image_file.read()))
            full_path = default_storage.path(saved_path)

            prompt, report = generate_road_damage_report(full_path)
            with open(full_path, 'rb') as f:
                predict_response = requests.post(
                    'http://127.0.0.1:8000/api/predict-classification/',
                    files={'image': f}
                )

            if predict_response.status_code == 200:
                prediction_data = predict_response.json()
            else:
                prediction_data = {
                    'error': 'Prediction service failed.',
                    'status_code': predict_response.status_code
                }

            report = report.replace("\n", " ")
            prompt = prompt.replace("\n", " ")
            default_storage.delete(saved_path)

            return Response({
                'prediction': prediction_data,
                'prompt': prompt,
                'report': report 
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
class VideoStreamDetectionView(APIView):
    """
    GET: Capture a frame from the stream, run YOLO, return detections and annotated image.
    """
    def get(self, request, *args, **kwargs):
        stream_url = "http://192.168.0.102:4747/video"  # Confirmed working

        cap = cv2.VideoCapture(stream_url)  # ← NO cv2.CAP_FFMPEG
        if not cap.isOpened():
            return Response(
                {"error": "Failed to open video stream"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return Response(
                {"error": "Failed to read frame from stream"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        # YOLOv8 inference
        results = model(frame)
        detections = []

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = results[0].names[cls]
            conf = float(box.conf[0])
            detections.append({
                "class": label,
                "confidence": round(conf, 2)
            })
        # Annotate image
        annotated_frame = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return Response({
            "status": "success",
            "detections": detections,
            #"annotated_image": img_base64
        }, status=status.HTTP_200_OK)
    

class VideoStreamMJPEGView(APIView):
    def get(self, request, *args, **kwargs):
        try:
            return StreamingHttpResponse(
                generate_stream(),
                content_type='multipart/x-mixed-replace; boundary=frame'
            )
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)





        # class Detect(APIView):
#     def post(self, request, *args, **kwargs):
#         image_file = request.FILES.get('image')  # The key must be 'image'

#         if not image_file:
#             return Response({'error': 'Image file not provided.'}, status=status.HTTP_400_BAD_REQUEST)

#         try:
#             # Save uploaded file to a temp location
#             file_ext = os.path.splitext(image_file.name)[-1]
#             file_name = f"{uuid.uuid4()}{file_ext}"
#             file_path = os.path.join('uploads', file_name)

#             saved_path = default_storage.save(file_path, ContentFile(image_file.read()))
#             full_path = default_storage.path(saved_path)

#             # Call your detection function
#             prompt, report = generate_road_damage_report(full_path)

#             # Optionally delete file after processing (recommended for temp files)
#             default_storage.delete(saved_path)

#             return Response({
#                 'prompt': prompt,
#                 'report': report
#             }, status=status.HTTP_200_OK)

#         except Exception as e:
#             return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
