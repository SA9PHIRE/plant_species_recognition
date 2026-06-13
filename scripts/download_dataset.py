"""
Скрипт для скачивания датасета с Kaggle.
Запуск: python scripts/download_dataset.py
"""

import os
import sys
from pathlib import Path

# Добавляем путь к корню проекта (чтобы работали импорты)
sys.path.append(str(Path(__file__).parent.parent))

# Попробуем разные библиотеки для скачивания
try:
    import kagglehub
    print("Используем kagglehub...")
except ImportError:
    print("Устанавливаем kagglehub...")
    os.system("pip install kagglehub")
    import kagglehub

def main():
    # ID датасета на Kaggle (например, "uciml/iris" или "yourname/dataset")
    DATASET_ID = "krushnamohod/plant-species-recognition"  # 🔁 ЗАМЕНИТЕ НА ВАШ
    
    # Куда сохраняем — используем path относительно корня проекта
    DOWNLOAD_PATH = Path(__file__).parent.parent / "data" / "raw"
    
    print(f"Скачиваем {DATASET_ID}...")
    print(f"Сохраняем в: {DOWNLOAD_PATH}")
    
    # Создаём папку, если её нет
    DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
    
    # Скачиваем
    try:
        path = kagglehub.dataset_download(DATASET_ID, path=str(DOWNLOAD_PATH))
        print(f"✅ Датасет успешно скачан в: {path}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("\nПроверьте:")
        print("1. Установлен ли kagglehub: pip install kagglehub")
        print("2. Настроен ли API ключ Kaggle (файл ~/.kaggle/kaggle.json)")
        print("3. Правильный ли ID датасета?")

if __name__ == "__main__":
    main()
