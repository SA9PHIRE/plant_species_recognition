"""
Модуль для загрузки и подготовки датасета с растениями
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np

class PlantDataset(Dataset):
    """Датасет для загрузки фотографий растений"""
    
    def __init__(self, root_dir: str, transform=None, is_train: bool = True):
        """
        Args:
            root_dir: Путь к папке с данными (например, data/raw/)
            transform: Преобразования для изображений
            is_train: True для обучения, False для валидации
        """
        self.root_dir = Path(root_dir)
        self.is_train = is_train
        
        # Собираем все изображения
        self.images = []
        self.labels = []
        self.classes = []
        
        # Проходим по всем папкам (каждая папка = вид растения)
        for class_idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if class_dir.is_dir():
                self.classes.append(class_dir.name)
                for img_path in class_dir.glob("*.jpg"):
                    self.images.append(img_path)
                    self.labels.append(class_idx)
        
        # Словарь для преобразования названия класса в индекс
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Преобразования для изображений
        if transform is None:
            self.transform = self._get_transforms()
        else:
            self.transform = transform
            
        print(f"📊 Загружено {len(self.images)} изображений")
        print(f"🌿 Найдено {len(self.classes)} видов растений: {', '.join(self.classes)}")
    
    def _get_transforms(self):
        """Определяет преобразования для изображений"""
        if self.is_train:
            # Для обучения: добавляем аугментацию (искусственное разнообразие)
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),      # Отражение
                transforms.RandomRotation(degrees=15),       # Поворот
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # Средние значения для ImageNet
                    std=[0.229, 0.224, 0.225]    # Стандартные отклонения
                )
            ])
        else:
            # Для валидации: только базовые преобразования
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Возвращает (изображение, метка)"""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Загружаем изображение
        image = Image.open(img_path).convert('RGB')
        
        # Применяем преобразования
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_dataloaders(
    data_dir: str, 
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Создаёт загрузчики данных для обучения и валидации
    
    Args:
        data_dir: Путь к папке с данными
        batch_size: Размер батча
        val_split: Доля данных для валидации (0-1)
        num_workers: Количество потоков для загрузки
    
    Returns:
        train_loader, val_loader, class_names
    """
    
    # Создаём полный датасет
    full_dataset = PlantDataset(data_dir, is_train=True)
    
    # Разделяем на train и val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Фиксируем случайность
    )
    
    # Меняем transform для валидационной части
    val_dataset.dataset.is_train = False
    val_dataset.dataset.transform = val_dataset.dataset._get_transforms()
    
    # Создаём загрузчики
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n📦 Разделение данных:")
    print(f"   Обучающая выборка: {train_size} изображений")
    print(f"   Валидационная: {val_size} изображений")
    print(f"   Размер батча: {batch_size}")
    
    return train_loader, val_loader, full_dataset.classes
