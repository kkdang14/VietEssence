# import os
# import torch
# import torch.nn as nn
# import json
# import uuid
# from datetime import datetime
# from PIL import Image
# import torchvision.transforms as transforms
# from django.http import JsonResponse
# from django.utils.decorators import method_decorator
# from django.views.decorators.csrf import csrf_exempt
# from rest_framework.views import APIView
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework import status
# import logging

# logger = logging.getLogger(__name__)

# # Disable OneDNN optimizations (if needed)
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# from fastkan import FastKAN
# import torchvision.models as models

# class ConvNeXtFastKAN(nn.Module):
#     def __init__(self, hidden_dims=None, num_classes=20, pretrained=True, freeze_backbone=True):
#         super(ConvNeXtFastKAN, self).__init__()
        
#         # Load pre-trained ConvNeXt model
#         self.convnext = models.convnext_base(weights=None if not pretrained else "DEFAULT")

#         # Freeze ConvNeXt layers if specified
#         if freeze_backbone:
#             for param in self.convnext.parameters():
#                 param.requires_grad = False

#         # Get the feature dimension from ConvNeXt
#         num_features = self.convnext.classifier[2].in_features
#         self.convnext.classifier = nn.Identity()  # Remove the classifier
        
#         # Default hidden dimensions if not provided
#         if hidden_dims is None:
#             hidden_dims = [512, 256]
        
#         # Create the complete layers list including input and output dimensions
#         layers_hidden = [num_features] + hidden_dims + [num_classes]
        
#         # Create FastKAN network with the specified architecture
#         self.fastkan = FastKAN(
#             layers_hidden=layers_hidden,
#             grid_min=-2.0,
#             grid_max=2.0,
#             num_grids=8,
#             use_base_update=True
#         )

#     def forward(self, x):
#         x = self.convnext(x)
#         x = x.view(x.size(0), -1)  # Flatten the tensor
#         x = self.fastkan(x)
#         return x

# #####################################
# # 2. Classification Results Storage
# #####################################

# class ClassificationStorage:
#     """Singleton class to manage classification results"""
#     _instance = None
#     _classification_results = []
#     _max_results = 1000  # Limit number of stored results
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super(ClassificationStorage, cls).__new__(cls)
#         return cls._instance
    
#     def add_result(self, result):
#         """Add new classification result to the list"""
#         result['id'] = str(uuid.uuid4())
#         result['created_at'] = datetime.now().isoformat()
        
#         self._classification_results.insert(0, result)  # Add to the beginning
        
#         # Limit number of results
#         if len(self._classification_results) > self._max_results:
#             self._classification_results = self._classification_results[:self._max_results]
    
#     def get_results(self, limit=None, offset=0):
#         """Get results with pagination"""
#         if limit is None:
#             return self._classification_results[offset:]
#         return self._classification_results[offset:offset + limit]
    
#     def get_result_by_id(self, result_id):
#         """Get result by ID"""
#         for result in self._classification_results:
#             if result['id'] == result_id:
#                 return result
#         return None
    
#     def delete_result(self, result_id):
#         """Delete result by ID"""
#         for i, result in enumerate(self._classification_results):
#             if result['id'] == result_id:
#                 self._classification_results.pop(i)
#                 return True
#         return False
    
#     def clear_all_results(self):
#         """Clear all results"""
#         self._classification_results.clear()
    
#     def get_statistics(self):
#         """Get statistics of results"""
#         total_results = len(self._classification_results)
        
#         # Statistics by class
#         class_stats = {}
#         confidence_stats = []
        
#         for result in self._classification_results:
#             if 'predictions' in result and result['predictions']:
#                 top_prediction = result['predictions'][0]
#                 class_name = top_prediction['class_name']
#                 confidence = top_prediction['confidence']
                
#                 if class_name not in class_stats:
#                     class_stats[class_name] = 0
#                 class_stats[class_name] += 1
#                 confidence_stats.append(confidence)
        
#         avg_confidence = sum(confidence_stats) / len(confidence_stats) if confidence_stats else 0
        
#         return {
#             'total_results': total_results,
#             'class_distribution': class_stats,
#             'average_confidence': round(avg_confidence, 4),
#             'max_confidence': max(confidence_stats) if confidence_stats else 0,
#             'min_confidence': min(confidence_stats) if confidence_stats else 0
#         }

# #####################################
# # 3. Model Loader
# #####################################

# class ModelLoader:
#     _instance = None
#     _model = None
#     _class_labels = None
#     _transform = None
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super(ModelLoader, cls).__new__(cls)
#         return cls._instance
    
#     def load_model(self, model_path, labels_path, num_classes, hidden_dims=None):
#         if self._model is None:
#             try:
#                 # Load model
#                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#                 self._model = ConvNeXtFastKAN(
#                     hidden_dims=hidden_dims,
#                     num_classes=num_classes,
#                     pretrained=True,
#                     freeze_backbone=True
#                 )
                
#                 # Load trained weights
#                 checkpoint = torch.load(model_path, map_location=device)
#                 if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#                     self._model.load_state_dict(checkpoint['model_state_dict'])
#                 elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#                     self._model.load_state_dict(checkpoint['state_dict'])
#                 else:
#                     self._model.load_state_dict(checkpoint)
                
#                 self._model.to(device)
#                 self._model.eval()
                
#                 # Load class labels
#                 with open(labels_path, 'r', encoding='utf-8') as f:
#                     self._class_labels = json.load(f)
                
#                 # Define image transforms
#                 self._transform = transforms.Compose([
#                     transforms.Resize((224, 224)),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                       std=[0.229, 0.224, 0.225])
#                 ])
                
#                 logger.info(f"ConvNeXtFastKAN model loaded successfully from {model_path}")
#                 logger.info(f"Model architecture: ConvNeXt backbone + FastKAN classifier")
#                 logger.info(f"Number of classes: {num_classes}")
                
#             except Exception as e:
#                 logger.error(f"Error loading ConvNeXtFastKAN model: {str(e)}")
#                 raise e
    
#     def get_model(self):
#         return self._model, self._class_labels, self._transform

# #####################################
# # 4. API Classification with List Management
# #####################################

# @method_decorator(csrf_exempt, name='dispatch')
# class ImageClassificationAPI(APIView):
#     parser_classes = (MultiPartParser, FormParser)
    
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.model_loader = ModelLoader()
#         self.storage = ClassificationStorage()
        
#         # Model configuration
#         self.model_path = os.path.join(os.environ.get('MODEL_DIR', 'models'), 'convnext_fastkan_best_model.pth')
#         self.labels_path = os.path.join(os.environ.get('MODEL_DIR', 'models'), 'class_labels.json')
#         self.num_classes = 11
#         self.hidden_dims = [512, 256]
        
#         # Load model
#         try:
#             self.model_loader.load_model(
#                 self.model_path, 
#                 self.labels_path, 
#                 self.num_classes,
#                 self.hidden_dims
#             )
#         except Exception as e:
#             logger.error(f"Failed to initialize ConvNeXtFastKAN model: {str(e)}")
    
#     def post(self, request, *args, **kwargs):
#         """Classify image and add to results list"""
#         try:
#             image_file = request.FILES.get('image', None)
#             top_k = int(request.data.get('top_k', 5))
#             save_to_list = request.data.get('save_to_list', 'true').lower() == 'true'
            
#             logger.info("Received image classification request")
            
#             if not image_file:
#                 return JsonResponse({
#                     "error": "No image file provided."
#                 }, status=status.HTTP_400_BAD_REQUEST)

#             # Process image
#             try:
#                 image = Image.open(image_file).convert('RGB')
#             except Exception as e:
#                 return JsonResponse({
#                     "error": f"Invalid image format: {str(e)}"
#                 }, status=status.HTTP_400_BAD_REQUEST)
            
#             # Get model
#             model, class_labels, transform = self.model_loader.get_model()
            
#             if model is None:
#                 return JsonResponse({
#                     "error": "Model not loaded properly."
#                 }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
#             # Preprocessing
#             try:
#                 input_tensor = transform(image).unsqueeze(0)
#                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#                 input_tensor = input_tensor.to(device)
#             except Exception as e:
#                 return JsonResponse({
#                     "error": f"Error preprocessing image: {str(e)}"
#                 }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
#             # Inference
#             try:
#                 with torch.no_grad():
#                     outputs = model(input_tensor)
#                     probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    
#                     top_probabilities, top_indices = torch.topk(probabilities, top_k)
                    
#                     top_probabilities = top_probabilities.cpu().numpy()
#                     top_indices = top_indices.cpu().numpy()
                    
#                     predictions = []
#                     for i in range(top_k):
#                         class_idx = int(top_indices[i])
#                         confidence = float(top_probabilities[i])
#                         class_name = class_labels.get(str(class_idx), f"Class_{class_idx}")
                        
#                         predictions.append({
#                             "class_id": class_idx,
#                             "class_name": class_name,
#                             "confidence": round(confidence, 4),
#                             "percentage": round(confidence * 100, 2)
#                         })
                    
#             except Exception as e:
#                 return JsonResponse({
#                     "error": f"Error during inference: {str(e)}"
#                 }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
#             # Create result
#             result_data = {
#                 "original_filename": image_file.name,
#                 "predictions": predictions,
#                 "top_prediction": predictions[0] if predictions else None,
#                 "model_info": {
#                     "device": str(device),
#                     "top_k": top_k
#                 }
#             }
            
#             # Add to list if requested
#             if save_to_list:
#                 self.storage.add_result(result_data.copy())
            
#             # Response
#             response_data = {
#                 "success": True,
#                 "result": result_data,
#                 "saved_to_list": save_to_list,
#                 "total_results_in_list": len(self.storage.get_results()),
#                 "timestamp": datetime.now().isoformat()
#             }
            
#             logger.info(f"Classification successful: {predictions[0]['class_name']} ({predictions[0]['percentage']}%)")
            
#             return JsonResponse(response_data, status=status.HTTP_200_OK)
            
#         except Exception as e:
#             logger.error(f"Unexpected error: {str(e)}")
#             return JsonResponse({
#                 "error": f"Internal server error: {str(e)}"
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
#     def get(self, request, *args, **kwargs):
#         """Get model info and statistics"""
#         try:
#             model, class_labels, _ = self.model_loader.get_model()
#             statistics = self.storage.get_statistics()
            
#             return JsonResponse({
#                 "success": True,
#                 "model_info": {
#                     "total_classes": len(class_labels) if class_labels else 0,
#                     "classes": class_labels,
#                     "model_loaded": model is not None,
#                     "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#                 },
#                 "statistics": statistics
#             }, status=status.HTTP_200_OK)
            
#         except Exception as e:
#             logger.error(f"Error getting model info: {str(e)}")
#             return JsonResponse({
#                 "error": f"Error getting model info: {str(e)}"
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# #####################################
# # 5. API for managing classification results
# #####################################

# @method_decorator(csrf_exempt, name='dispatch')
# class ClassificationResultsAPI(APIView):
#     """API to manage classification results list"""
    
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.storage = ClassificationStorage()
    
#     def get(self, request, *args, **kwargs):
#         """Get results list with pagination"""
#         try:
#             limit = request.GET.get('limit', None)
#             offset = int(request.GET.get('offset', 0))
            
#             if limit is not None:
#                 limit = int(limit)
            
#             results = self.storage.get_results(limit=limit, offset=offset)
#             total_count = len(self.storage.get_results())
#             statistics = self.storage.get_statistics()
            
#             return JsonResponse({
#                 "success": True,
#                 "results": results,
#                 "pagination": {
#                     "total_count": total_count,
#                     "offset": offset,
#                     "limit": limit,
#                     "returned_count": len(results)
#                 },
#                 "statistics": statistics
#             }, status=status.HTTP_200_OK)
            
#         except Exception as e:
#             logger.error(f"Error getting results: {str(e)}")
#             return JsonResponse({
#                 "error": f"Error getting results: {str(e)}"
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
#     def delete(self, request, *args, **kwargs):
#         """Delete result by ID or clear all"""
#         try:
#             result_id = request.GET.get('id', None)
#             clear_all = request.GET.get('clear_all', 'false').lower() == 'true'
            
#             if clear_all:
#                 self.storage.clear_all_results()
#                 return JsonResponse({
#                     "success": True,
#                     "message": "All results cleared successfully"
#                 }, status=status.HTTP_200_OK)
            
#             elif result_id:
#                 success = self.storage.delete_result(result_id)
#                 if success:
#                     return JsonResponse({
#                         "success": True,
#                         "message": f"Result {result_id} deleted successfully"
#                     }, status=status.HTTP_200_OK)
#                 else:
#                     return JsonResponse({
#                         "error": f"Result {result_id} not found"
#                     }, status=status.HTTP_404_NOT_FOUND)
            
#             else:
#                 return JsonResponse({
#                     "error": "Please provide 'id' parameter or set 'clear_all=true'"
#                 }, status=status.HTTP_400_BAD_REQUEST)
                
#         except Exception as e:
#             logger.error(f"Error deleting results: {str(e)}")
#             return JsonResponse({
#                 "error": f"Error deleting results: {str(e)}"
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# @method_decorator(csrf_exempt, name='dispatch')
# class ClassificationResultDetailAPI(APIView):
#     """API to get details of a single classification result"""
    
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.storage = ClassificationStorage()
    
#     def get(self, request, result_id, *args, **kwargs):
#         """Get detailed result by ID"""
#         try:
#             result = self.storage.get_result_by_id(result_id)
            
#             if result:
#                 return JsonResponse({
#                     "success": True,
#                     "result": result
#                 }, status=status.HTTP_200_OK)
#             else:
#                 return JsonResponse({
#                     "error": f"Result {result_id} not found"
#                 }, status=status.HTTP_404_NOT_FOUND)
                
#         except Exception as e:
#             logger.error(f"Error getting result detail: {str(e)}")
#             return JsonResponse({
#                 "error": f"Error getting result detail: {str(e)}"
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

import os
import torch
import torch.nn as nn
import json
import uuid
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
import logging

logger = logging.getLogger(__name__)

# Disable OneDNN optimizations (if needed)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastkan import FastKAN
import torchvision.models as models

class ConvNeXtFastKAN(nn.Module):
    def __init__(self, hidden_dims=None, num_classes=20, pretrained=True, freeze_backbone=True):
        super(ConvNeXtFastKAN, self).__init__()
        
        # Load pre-trained ConvNeXt model
        self.convnext = models.convnext_base(weights=None if not pretrained else "DEFAULT")

        # Freeze ConvNeXt layers if specified
        if freeze_backbone:
            for param in self.convnext.parameters():
                param.requires_grad = False

        # Get the feature dimension from ConvNeXt
        num_features = self.convnext.classifier[2].in_features
        self.convnext.classifier = nn.Identity()  # Remove the classifier
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Create the complete layers list including input and output dimensions
        layers_hidden = [num_features] + hidden_dims + [num_classes]
        
        # Create FastKAN network with the specified architecture
        self.fastkan = FastKAN(
            layers_hidden=layers_hidden,
            grid_min=-2.0,
            grid_max=2.0,
            num_grids=8,
            use_base_update=True
        )

    def forward(self, x):
        x = self.convnext(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fastkan(x)
        return x

#####################################
# 2. Classification Results Storage
#####################################

class ClassificationStorage:
    """Singleton class to manage classification results"""
    _instance = None
    _classification_results = []
    _max_results = 1000  # Limit number of stored results
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClassificationStorage, cls).__new__(cls)
        return cls._instance
    
    def add_result(self, result):
        """Add new classification result to the list"""
        result['id'] = str(uuid.uuid4())
        result['created_at'] = datetime.now().isoformat()
        
        self._classification_results.insert(0, result)  # Add to the beginning
        
        # Limit number of results
        if len(self._classification_results) > self._max_results:
            self._classification_results = self._classification_results[:self._max_results]
    
    def get_results(self, limit=None, offset=0):
        """Get results with pagination"""
        if limit is None:
            return self._classification_results[offset:]
        return self._classification_results[offset:offset + limit]
    
    def get_result_by_id(self, result_id):
        """Get result by ID"""
        for result in self._classification_results:
            if result['id'] == result_id:
                return result
        return None
    
    def delete_result(self, result_id):
        """Delete result by ID"""
        for i, result in enumerate(self._classification_results):
            if result['id'] == result_id:
                self._classification_results.pop(i)
                return True
        return False
    
    def clear_all_results(self):
        """Clear all results"""
        self._classification_results.clear()
    
    def get_statistics(self):
        """Get statistics of results"""
        total_results = len(self._classification_results)
        
        # Statistics by class
        class_stats = {}
        confidence_stats = []
        
        for result in self._classification_results:
            if 'predictions' in result and result['predictions']:
                top_prediction = result['predictions'][0]
                class_name = top_prediction['class_name']
                confidence = top_prediction['confidence']
                
                if class_name not in class_stats:
                    class_stats[class_name] = 0
                class_stats[class_name] += 1
                confidence_stats.append(confidence)
        
        avg_confidence = sum(confidence_stats) / len(confidence_stats) if confidence_stats else 0
        
        return {
            'total_results': total_results,
            'class_distribution': class_stats,
            'average_confidence': round(avg_confidence, 4),
            'max_confidence': max(confidence_stats) if confidence_stats else 0,
            'min_confidence': min(confidence_stats) if confidence_stats else 0
        }

#####################################
# 3. Model Loader
#####################################

class ModelLoader:
    _instance = None
    _model = None
    _class_labels = None
    _transform = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path, labels_path, num_classes, hidden_dims=None):
        if self._model is None:
            try:
                # Determine device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                logger.info(f"Using device: {device}")

                # Load model
                self._model = ConvNeXtFastKAN(
                    hidden_dims=hidden_dims,
                    num_classes=num_classes,
                    pretrained=True,
                    freeze_backbone=True
                )
                
                # Load trained weights
                checkpoint = torch.load(model_path, map_location=device)
                
                # Handle state_dict key mismatch (remove 'module.' prefix)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove 'module.' prefix from keys if present
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace('module.', '') if key.startswith('module.') else key
                    new_state_dict[new_key] = value
                
                # Load the modified state_dict
                self._model.load_state_dict(new_state_dict)
                
                self._model.to(device)
                self._model.eval()
                
                # Load class labels
                with open(labels_path, 'r', encoding='utf-8') as f:
                    self._class_labels = json.load(f)
                
                # Define image transforms
                self._transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
                ])
                
                logger.info(f"ConvNeXtFastKAN model loaded successfully from {model_path}")
                logger.info(f"Model architecture: ConvNeXt backbone + FastKAN classifier")
                logger.info(f"Number of classes: {num_classes}")
                
            except Exception as e:
                logger.error(f"Error loading ConvNeXtFastKAN model: {str(e)}")
                raise e
    
    def get_model(self):
        return self._model, self._class_labels, self._transform

#####################################
# 4. API Classification with List Management
#####################################

@method_decorator(csrf_exempt, name='dispatch')
class ImageClassificationAPI(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_loader = ModelLoader()
        self.storage = ClassificationStorage()
        
        # Model configuration
        self.model_path = os.path.join(os.environ.get('MODEL_DIR', 'models'), 'convnext_fastkan_best_model.pth')
        self.labels_path = os.path.join(os.environ.get('MODEL_DIR', 'models'), 'class_labels.json')
        self.num_classes = 11
        self.hidden_dims = [512, 256]
        
        # Load model
        try:
            self.model_loader.load_model(
                self.model_path, 
                self.labels_path, 
                self.num_classes,
                self.hidden_dims
            )
        except Exception as e:
            logger.error(f"Failed to initialize ConvNeXtFastKAN model: {str(e)}")
    
    def post(self, request, *args, **kwargs):
        """Classify image and add to results list"""
        try:
            image_file = request.FILES.get('image', None)
            top_k = int(request.data.get('top_k', 5))
            save_to_list = request.data.get('save_to_list', 'true').lower() == 'true'
            
            logger.info("Received image classification request")
            
            if not image_file:
                return JsonResponse({
                    "error": "No image file provided."
                }, status=status.HTTP_400_BAD_REQUEST)

            # Process image
            try:
                image = Image.open(image_file).convert('RGB')
            except Exception as e:
                return JsonResponse({
                    "error": f"Invalid image format: {str(e)}"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Get model
            model, class_labels, transform = self.model_loader.get_model()
            
            if model is None:
                return JsonResponse({
                    "error": "Model not loaded properly."
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Preprocessing
            try:
                input_tensor = transform(image).unsqueeze(0)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                input_tensor = input_tensor.to(device)
            except Exception as e:
                return JsonResponse({
                    "error": f"Error preprocessing image: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Inference
            try:
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    
                    top_probabilities, top_indices = torch.topk(probabilities, top_k)
                    
                    top_probabilities = top_probabilities.cpu().numpy()
                    top_indices = top_indices.cpu().numpy()
                    
                    predictions = []
                    for i in range(top_k):
                        class_idx = int(top_indices[i])
                        confidence = float(top_probabilities[i])
                        class_name = class_labels.get(str(class_idx), f"Class_{class_idx}")
                        
                        predictions.append({
                            "class_id": class_idx,
                            "class_name": class_name,
                            "confidence": round(confidence, 4),
                            "percentage": round(confidence * 100, 2)
                        })
                    
            except Exception as e:
                return JsonResponse({
                    "error": f"Error during inference: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Create result
            result_data = {
                "original_filename": image_file.name,
                "predictions": predictions,
                "top_prediction": predictions[0] if predictions else None,
                "model_info": {
                    "device": str(device),
                    "top_k": top_k
                }
            }
            
            # Add to list if requested
            if save_to_list:
                self.storage.add_result(result_data.copy())
            
            # Response
            response_data = {
                "success": True,
                "result": result_data,
                "saved_to_list": save_to_list,
                "total_results_in_list": len(self.storage.get_results()),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Classification successful: {predictions[0]['class_name']} ({predictions[0]['percentage']}%)")
            
            return JsonResponse(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return JsonResponse({
                "error": f"Internal server error: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def get(self, request, *args, **kwargs):
        """Get model info and statistics"""
        try:
            model, class_labels, _ = self.model_loader.get_model()
            statistics = self.storage.get_statistics()
            
            return JsonResponse({
                "success": True,
                "model_info": {
                    "total_classes": len(class_labels) if class_labels else 0,
                    "classes": class_labels,
                    "model_loaded": model is not None,
                    "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                },
                "statistics": statistics
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return JsonResponse({
                "error": f"Error getting model info: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#####################################
# 5. API for managing classification results
#####################################

@method_decorator(csrf_exempt, name='dispatch')
class ClassificationResultsAPI(APIView):
    """API to manage classification results list"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage = ClassificationStorage()
    
    def get(self, request, *args, **kwargs):
        """Get results list with pagination"""
        try:
            limit = request.GET.get('limit', None)
            offset = int(request.GET.get('offset', 0))
            
            if limit is not None:
                limit = int(limit)
            
            results = self.storage.get_results(limit=limit, offset=offset)
            total_count = len(self.storage.get_results())
            statistics = self.storage.get_statistics()
            
            return JsonResponse({
                "success": True,
                "results": results,
                "pagination": {
                    "total_count": total_count,
                    "offset": offset,
                    "limit": limit,
                    "returned_count": len(results)
                },
                "statistics": statistics
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error getting results: {str(e)}")
            return JsonResponse({
                "error": f"Error getting results: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def delete(self, request, *args, **kwargs):
        """Delete result by ID or clear all"""
        try:
            result_id = request.GET.get('id', None)
            clear_all = request.GET.get('clear_all', 'false').lower() == 'true'
            
            if clear_all:
                self.storage.clear_all_results()
                return JsonResponse({
                    "success": True,
                    "message": "All results cleared successfully"
                }, status=status.HTTP_200_OK)
            
            elif result_id:
                success = self.storage.delete_result(result_id)
                if success:
                    return JsonResponse({
                        "success": True,
                        "message": f"Result {result_id} deleted successfully"
                    }, status=status.HTTP_200_OK)
                else:
                    return JsonResponse({
                        "error": f"Result {result_id} not found"
                    }, status=status.HTTP_404_NOT_FOUND)
            
            else:
                return JsonResponse({
                    "error": "Please provide 'id' parameter or set 'clear_all=true'"
                }, status=status.HTTP_400_BAD_REQUEST)
                
        except Exception as e:
            logger.error(f"Error deleting results: {str(e)}")
            return JsonResponse({
                "error": f"Error deleting results: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@method_decorator(csrf_exempt, name='dispatch')
class ClassificationResultDetailAPI(APIView):
    """API to get details of a single classification result"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.storage = ClassificationStorage()
    
    def get(self, request, result_id, *args, **kwargs):
        """Get detailed result by ID"""
        try:
            result = self.storage.get_result_by_id(result_id)
            
            if result:
                return JsonResponse({
                    "success": True,
                    "result": result
                }, status=status.HTTP_200_OK)
            else:
                return JsonResponse({
                    "error": f"Result {result_id} not found"
                }, status=status.HTTP_404_NOT_FOUND)
                
        except Exception as e:
            logger.error(f"Error getting result detail: {str(e)}")
            return JsonResponse({
                "error": f"Error getting result detail: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)