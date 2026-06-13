"""
Предсказание вида растения на фото
Запуск: python src/predict.py --image путь/к/фото.jpg
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
from torchvision import transforms
from PIL import Image
import argparse

from src.model import create_model

def load_class_names(class_file: Path):
    """Загружает названия классов"""
    classes = {}
    with open(class_file, 'r') as f:
        for line in f:
            idx, name = line.strip().split(': ')
            classes[int(idx)] = name
    return classes

def predict_image(model, image_path, class_names, device):
    """Предсказывает вид растения"""
    
    # Преобразования
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Загружаем и преобразуем изображение
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Предсказание
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.topk(probabilities, 3)
    
    # Выводим результаты
    print(f"\n🌿 Анализ изображения: {image_path.name}")
    print("=" * 40)
    
    for i in range(3):
        class_idx = top_class[0][i].item()
        prob = top_prob[0][i].item() * 100
        print(f"{i+1}. {class_names[class_idx]}: {prob:.2f}%")
    
    return top_class[0][0].item()

def main():
    parser = argparse.ArgumentParser(description='Распознавание растений')
    parser.add_argument('--image', type=str, required=True, help='Путь к фото')
    parser.add_argument('--model', type=str, default='models/best_model_epoch_*.pth', 
                        help='Путь к модели')
    args = parser.parse_args()
    
    # Настройки
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Загружаем классы
    class_file = Path('models/classes.txt')
    if not class_file.exists():
        print("❌ Файл classes.txt не найден! Сначала обучите модель.")
        return
    
    class_names = load_class_names(class_file)
    num_classes = len(class_names)
    
    # Загружаем модель
    model = create_model(num_classes=num_classes, device=device)
    
    # Ищем последнюю модель
    model_dir = Path('models')
    model_files = list(model_dir.glob('best_model_epoch_*.pth'))
    if model_files:
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        print(f"📂 Загружаем модель: {latest_model}")
        checkpoint = torch.load(latest_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("❌ Модель не найдена! Сначала обучите модель.")
        return
    
    # Предсказываем
    predict_image(model, args.image, class_names, device)

if __name__ == "__main__":
    main()
