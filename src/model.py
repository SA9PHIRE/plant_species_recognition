"""
Архитектура нейросети для распознавания растений
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict

class PlantClassifier(nn.Module):
    """Классификатор растений на основе предобученной ResNet-18"""
    
    def __init__(self, num_classes: int, model_name: str = "resnet18"):
        """
        Args:
            num_classes: Количество видов растений
            model_name: Название архитектуры ('resnet18', 'resnet50')
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Загружаем предобученную модель
        if model_name == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        elif model_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Неизвестная модель: {model_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через сеть"""
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Получить признаки перед последним слоем (для визуализации)"""
        # Убираем последний слой классификации
        features = self.backbone(x)
        return features
    
    def count_parameters(self) -> Dict[str, int]:
        """Подсчёт количества обучаемых параметров"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": total_params
        }

def create_model(num_classes: int, device: str = "cuda") -> nn.Module:
    """Создаёт и перемещает модель на устройство"""
    model = PlantClassifier(num_classes=num_classes, model_name="resnet18")
    model = model.to(device)
    
    print(f"\n🤖 Модель: {model.model_name}")
    print(f"   Количество параметров: {model.count_parameters()['total_parameters']:,}")
    print(f"   Устройство: {device}")
    
    return model
