"""
Вспомогательные функции для обучения
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple

def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int
) -> float:
    """Обучает модель одну эпоху"""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Прогресс-бар
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Обнуляем градиенты
        optimizer.zero_grad()
        
        # Прямой проход
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Обратный проход
        loss.backward()
        optimizer.step()
        
        # Статистика
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Обновляем прогресс-бар
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str
) -> Tuple[float, float]:
    """Валидация модели"""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    val_loss: float,
    filepath: Path
):
    """Сохраняет чекпоинт модели"""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }
    
    torch.save(checkpoint, filepath)
    print(f"💾 Чекпоинт сохранён: {filepath}")

def load_checkpoint(model: torch.nn.Module, filepath: Path, device: str):
    """Загружает чекпоинт модели"""
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"📂 Загружена модель: {filepath}")
    print(f"   Эпоха: {checkpoint['epoch']}, Accuracy: {checkpoint['val_acc']:.2f}%")
    
    return checkpoint

def plot_training_history(history: Dict[str, List[float]]):
    """Рисует графики обучения"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # График потерь
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Динамика потерь')
    ax1.legend()
    ax1.grid(True)
    
    # График точности
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Динамика точности')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
    print("📊 Графики сохранены в training_history.png")
