"""
Главный скрипт для обучения модели распознавания растений
Запуск: python src/train.py
"""

import sys
from pathlib import Path

# Добавляем корень проекта в пути поиска
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from datetime import datetime

# Импортируем наши модули
from src.dataset import create_dataloaders
from src.model import create_model
from src.utils import train_one_epoch, validate, save_checkpoint, plot_training_history

def main():
    print("=" * 60)
    print("🌿 РАСПОЗНАВАНИЕ ВИДОВ РАСТЕНИЙ - ОБУЧЕНИЕ НЕЙРОСЕТИ")
    print("=" * 60)
    
    # ==================== НАСТРОЙКИ ====================
    CONFIG = {
        # Данные
        'data_dir': 'data/raw',           # Папка с фотографиями
        'batch_size': 32,                 # Размер батча
        'val_split': 0.2,                 # Доля валидации
        
        # Обучение
        'num_epochs': 20,                 # Количество эпох
        'learning_rate': 0.001,           # Скорость обучения
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Сохранение
        'checkpoint_dir': Path('models'),
    }
    
    print(f"\n⚙️ Конфигурация обучения:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    # Создаём папку для сохранения моделей
    CONFIG['checkpoint_dir'].mkdir(exist_ok=True)
    
    # ==================== ЗАГРУЗКА ДАННЫХ ====================
    print("\n" + "=" * 60)
    print("📸 ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)
    
    try:
        train_loader, val_loader, class_names = create_dataloaders(
            data_dir=CONFIG['data_dir'],
            batch_size=CONFIG['batch_size'],
            val_split=CONFIG['val_split']
        )
    except FileNotFoundError:
        print("\n❌ ОШИБКА: Папка с данными не найдена!")
        print("Сначала скачайте датасет:")
        print("   python scripts/download_dataset.py")
        return
    
    # ==================== СОЗДАНИЕ МОДЕЛИ ====================
    print("\n" + "=" * 60)
    print("🤖 СОЗДАНИЕ НЕЙРОСЕТИ")
    print("=" * 60)
    
    model = create_model(
        num_classes=len(class_names),
        device=CONFIG['device']
    )
    
    # ==================== НАСТРОЙКА ОБУЧЕНИЯ ====================
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    # ==================== ОБУЧЕНИЕ ====================
    print("\n" + "=" * 60)
    print("🚀 НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 60)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Эпоха {epoch}/{CONFIG['num_epochs']}")
        print(f"{'='*50}")
        
        # Обучение
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, 
            CONFIG['device'], epoch
        )
        
        # Валидация
        val_loss, val_acc = validate(
            model, val_loader, criterion, CONFIG['device']
        )
        
        # Обновляем learning rate
        scheduler.step(val_loss)
        
        # Сохраняем историю
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Вывод результатов
        print(f"\n📊 Результаты эпохи {epoch}:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = CONFIG['checkpoint_dir'] / f"best_model_epoch_{epoch}.pth"
            save_checkpoint(
                model, optimizer, epoch, val_acc, val_loss, checkpoint_path
            )
            print(f"🎉 Новая лучшая модель! Accuracy: {val_acc:.2f}%")
    
    # ==================== ИТОГИ ====================
    print("\n" + "=" * 60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    
    print(f"\n📈 Лучшая точность на валидации: {best_val_acc:.2f}%")
    print(f"💾 Модели сохранены в папке: {CONFIG['checkpoint_dir']}")
    
    # Сохраняем информацию о классах
    with open(CONFIG['checkpoint_dir'] / 'classes.txt', 'w') as f:
        for idx, class_name in enumerate(class_names):
            f.write(f"{idx}: {class_name}\n")
    
    # Рисуем графики
    plot_training_history(history)
    
    print("\n🎯 Чтобы использовать модель для предсказаний, запустите:")
    print("   python src/predict.py --image путь/к/фото.jpg")

if __name__ == "__main__":
    main()
