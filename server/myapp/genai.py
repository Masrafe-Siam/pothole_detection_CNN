from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import uuid
import requests

from .generative import generate_road_damage_report

class Detect(APIView):
    def post(self, request, *args, **kwargs):
        image_file = request.FILES.get('image')

        if not image_file:
            return Response({'error': 'Image file not provided.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Save uploaded file
            file_ext = os.path.splitext(image_file.name)[-1]
            file_name = f"{uuid.uuid4()}{file_ext}"
            file_path = os.path.join('uploads', file_name)

            saved_path = default_storage.save(file_path, ContentFile(image_file.read()))
            full_path = default_storage.path(saved_path)

            # Generate damage report
            prompt, report = generate_road_damage_report(full_path)

            # Call internal prediction endpoint
            with open(full_path, 'rb') as f:
                predict_response = requests.post(
                    'http://127.0.0.1:8000/api/predict/',
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
