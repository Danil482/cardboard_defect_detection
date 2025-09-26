import os
import random
import json

# Параметры
K = 20  # Количество бутстрэп-подвыборок
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

IMAGES_DIR = os.path.join("dataset_VOC", "JPEGImages")  # Путь к изображениям
LABELS_DIR = "labels"  # Путь к меткам

# Директория для сохранения списков подвыборок
BOOTSTRAP_LISTS_DIR = "bootstrap_lists"
os.makedirs(BOOTSTRAP_LISTS_DIR, exist_ok=True)

def main():
    # 1. Соберём список всех файлов изображений
    all_images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Всего изображений: {len(all_images)}")

    # 2. Генерируем K бутстрэп-подвыборок с пропорцией 7:2:1
    total_count = len(all_images)
    train_count = int(total_count * 0.7)  # 739
    val_count = int(total_count * 0.2)    # 211
    test_count = total_count - train_count - val_count  # 106

    print(f"Целевое разбиение для каждой подвыборки: train={train_count}, val={val_count}, test={test_count}")

    for i in range(K):
        # Бутстрэп с возвращением: выбираем total_count файлов из всего датасета
        sampled_files = random.choices(all_images, k=total_count)

        # Перетасовываем для случайного разбиения
        random.shuffle(sampled_files)

        # Разбиваем на train, val, test в пропорции 7:2:1
        train_files = sampled_files[:train_count]
        val_files = sampled_files[train_count:train_count + val_count]
        test_files = sampled_files[train_count + val_count:]

        # Сохраняем списки в JSON
        bootstrap_data = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        json_path = os.path.join(BOOTSTRAP_LISTS_DIR, f"bootstrap_lists_{i}.json")
        with open(json_path, 'w') as f:
            json.dump(bootstrap_data, f, indent=4)
        print(f"Сгенерирована подвыборка {i+1}/{K}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test. Сохранено в {json_path}")

    print("\nГенерация подвыборок завершена. Используйте эти JSON-файлы для создания временных датасетов в скрипте обучения.")

if __name__ == '__main__':
    main()