from rest_framework import serializers
from .models import Image_Predictions


class ImagePredictionsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image_Predictions
        fields = '__all__'