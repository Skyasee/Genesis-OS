import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPProcessor, CLIPModel
from transformers import ViTFeatureExtractor, ViTModel
from ultralytics import YOLO
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
from PIL import Image
import cv2

class VisionPerception:
    """Vision perception system combining multiple models for comprehensive game understanding."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self._init_models()
        
        # Initialize feature extractors
        self._init_feature_extractors()
        
        # Cache for storing recent frames
        self.frame_buffer = []
        self.max_buffer_size = 30  # Store last 30 frames
        
    def _init_models(self):
        """Initialize all vision models."""
        try:
            # YOLO for object detection
            self.yolo = YOLO('yolov8n.pt')
            
            # CLIP for image-text understanding
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            
            # ViT for global context
            self.vit_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(self.device)
            
            # Custom CNN for game-specific features
            self.game_cnn = self._create_game_cnn()
            
            self.logger.info("Vision models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing vision models: {str(e)}")
            raise
    
    def _init_feature_extractors(self):
        """Initialize feature extraction layers."""
        self.feature_extractors = {
            'yolo': self._extract_yolo_features,
            'clip': self._extract_clip_features,
            'vit': self._extract_vit_features,
            'cnn': self._extract_cnn_features
        }
    
    def _create_game_cnn(self) -> nn.Module:
        """Create a custom CNN for game-specific feature extraction."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512)
        ).to(self.device)
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single game frame through all vision models."""
        try:
            # Convert frame to PIL Image for CLIP and ViT
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Update frame buffer
            self._update_frame_buffer(frame)
            
            # Extract features from all models
            features = {}
            for model_name, extractor in self.feature_extractors.items():
                features[model_name] = extractor(frame, pil_image)
            
            # Combine features
            combined_features = self._combine_features(features)
            
            return {
                'features': combined_features,
                'detections': features['yolo'],
                'frame_buffer': self.frame_buffer
            }
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return {}
    
    def _update_frame_buffer(self, frame: np.ndarray):
        """Update the frame buffer with the latest frame."""
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
    
    def _extract_yolo_features(self, frame: np.ndarray, _: Image.Image) -> List[Dict]:
        """Extract object detection features using YOLO."""
        results = self.yolo(frame)
        return [{
            'class': result.boxes.cls.tolist(),
            'confidence': result.boxes.conf.tolist(),
            'boxes': result.boxes.xyxy.tolist()
        } for result in results]
    
    def _extract_clip_features(self, _: np.ndarray, image: Image.Image) -> torch.Tensor:
        """Extract features using CLIP."""
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        return image_features
    
    def _extract_vit_features(self, _: np.ndarray, image: Image.Image) -> torch.Tensor:
        """Extract features using ViT."""
        inputs = self.vit_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
        return outputs.last_hidden_state
    
    def _extract_cnn_features(self, frame: np.ndarray, _: Image.Image) -> torch.Tensor:
        """Extract features using custom CNN."""
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.game_cnn(frame_tensor)
        return features
    
    def _combine_features(self, features: Dict[str, Any]) -> torch.Tensor:
        """Combine features from different models."""
        # Concatenate all feature vectors
        feature_vectors = []
        
        # Add CLIP features
        if 'clip' in features:
            feature_vectors.append(features['clip'].squeeze())
        
        # Add ViT features (use mean pooling)
        if 'vit' in features:
            feature_vectors.append(features['vit'].mean(dim=1).squeeze())
        
        # Add CNN features
        if 'cnn' in features:
            feature_vectors.append(features['cnn'].squeeze())
        
        # Concatenate all features
        combined = torch.cat(feature_vectors)
        
        # Normalize
        combined = torch.nn.functional.normalize(combined, p=2, dim=0)
        
        return combined
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get the current game state based on vision analysis."""
        if not self.frame_buffer:
            return {}
        
        # Process the most recent frame
        latest_frame = self.frame_buffer[-1]
        return self.process_frame(latest_frame)
    
    def save_state(self, path: str) -> None:
        """Save the current vision models state."""
        state = {
            'game_cnn': self.game_cnn.state_dict(),
            'config': self.config
        }
        torch.save(state, path)
    
    def load_state(self, path: str) -> None:
        """Load a previously saved vision models state."""
        state = torch.load(path)
        self.game_cnn.load_state_dict(state['game_cnn'])
        self.config = state['config'] 