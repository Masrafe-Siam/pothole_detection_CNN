# api/views.py
import numpy as np
import tensorflow as tf
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from PIL import Image
import io
import cv2

IMAGE_SIZE = 224
CLASS_NAMES = ['Crack', 'Pothole', 'Surface Erosion']

# # Load model once on server start
model = tf.keras.models.load_model("E:/python/ML/pothole_detection_CNN/Model/Trained Modelv2/MOBILENET.keras")
# MODEL_PATH = config("MODEL_PATH")  # Will load from .env
# model = tf.keras.models.load_model(MODEL_PATH)

def get_lighting_condition(img_array):
    grayscale_img = tf.image.rgb_to_grayscale(img_array)
    brightness = tf.reduce_mean(grayscale_img).numpy()
    if brightness < 50:
        lighting = "Very Low Light"
    elif brightness < 100:
        lighting = "Low Light"
    elif brightness < 170:
        lighting = "Moderate Light"
    elif brightness < 220:
        lighting = "Bright Light"
    else:
        lighting = "Very Bright Light"

    return lighting, round(brightness, 2)

from tensorflow.keras.applications.vgg19 import preprocess_input

def predict_image(model, img):
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    #img_array = preprocess_input(img_array)
    img_array_expanded = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array_expanded, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = round(100 * np.max(prediction[0]), 2)

    lighting, brightness = get_lighting_condition(img_array)
    return predicted_class, confidence, lighting, brightness


class PredictView(APIView):
    parser_classes = [MultiPartParser]
    def post(self, request):
        file_obj = request.FILES.get('image')
        if not file_obj:
            return Response({'error': 'No image provided'}, status=400)
        try:
            img = Image.open(io.BytesIO(file_obj.read())).convert('RGB')
            predicted_class, confidence, lighting, brightness = predict_image(model, img)
            return Response({
                'prediction': predicted_class,
                'confidence': confidence,
                'lighting': lighting,
                'brightness': brightness
            })
        except Exception as e:
            return Response({'error': str(e)}, status=500)



# import os
# import numpy as np
# import tensorflow as tf
# from PIL import Image
# import io

# from rest_framework.views import APIView
# from rest_framework.parsers import MultiPartParser
# from rest_framework.response import Response

# # Constants
# IMAGE_SIZE = 224
# CLASS_NAMES = ['Crack', 'Pothole', 'Surface Erosion']

# # Load model once at startup
# MODEL_PATH = "E:/python/ML/pothole_detection_CNN/Model/Trained Modelv2/CNN.h5"
# model = tf.keras.models.load_model(MODEL_PATH)

# # Auto preprocessing based on model filename
# def auto_preprocess(img_array, model_path):
#     model_name = os.path.basename(model_path).lower()
#     print(model_name)
#     if 'mobilenet' in model_name:
#         from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#         return preprocess_input(img_array)
#     elif 'vgg' in model_name:
#         from tensorflow.keras.applications.vgg19 import preprocess_input
#         return preprocess_input(img_array)
#     else:
#         return img_array / 255.0  # default fallback

# # Lighting detection
# def get_lighting_condition(img_array):
#     grayscale = tf.image.rgb_to_grayscale(img_array)
#     brightness = tf.reduce_mean(grayscale).numpy()

#     if brightness < 50:
#         lighting = "Very Low Light"
#     elif brightness < 100:
#         lighting = "Low Light"
#     elif brightness < 170:
#         lighting = "Moderate Light"
#     elif brightness < 220:
#         lighting = "Bright Light"
#     else:
#         lighting = "Very Bright Light"

#     return lighting, round(brightness, 2)

# # Prediction logic
# def predict_image(model, img):
#     img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     lighting, brightness = get_lighting_condition(img_array)

#     img_array = auto_preprocess(img_array, MODEL_PATH)
#     img_array_expanded = tf.expand_dims(img_array, 0)

#     prediction = model.predict(img_array_expanded, verbose=0)
#     predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
#     confidence = round(100 * np.max(prediction[0]), 2)

#     return predicted_class, confidence, lighting, brightness

# # API View
# class PredictView(APIView):
#     parser_classes = [MultiPartParser]

#     def post(self, request):
#         file_obj = request.FILES.get('image')
#         if not file_obj:
#             return Response({'error': 'No image provided'}, status=400)
#         try:
#             img = Image.open(io.BytesIO(file_obj.read())).convert('RGB')
#             predicted_class, confidence, lighting, brightness = predict_image(model, img)

#             return Response({
#                 'prediction': predicted_class,
#                 'confidence': confidence,
#                 'lighting': lighting,
#                 'brightness': brightness
#             })
#         except Exception as e:
#             return Response({'error': str(e)}, status=500)
