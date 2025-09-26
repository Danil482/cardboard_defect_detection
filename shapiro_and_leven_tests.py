import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# Путь для сохранения гистограмм
output_dir = 'master_work/Images'
os.makedirs(output_dir, exist_ok=True)

# Загрузка данных из CSV-файлов
files = {
    'yolov9t': 'bootstrap_results_yolov9t.csv',
    'yolo11n': 'bootstrap_results_yolo11n.csv',
    'yolov10n': 'bootstrap_results_yolov10n.csv',
    'yolov8n': 'bootstrap_results_yolov8n.csv',
    'yolov8n_tuned': 'bootstrap_results_yolov8n_tuned.csv'
}

mAP50_groups = {}
for model, file in files.items():
    df = pd.read_csv(file)
    mAP50_groups[model] = df['mAP50'].values  # Извлекаем mAP50 как numpy array

# 1. Тест Шапиро-Уилка для каждой группы
print("Тест Шапиро-Уилка (нормальность распределений):")
shapiro_results = {}
for model, data in mAP50_groups.items():
    stat, p = stats.shapiro(data)
    shapiro_results[model] = (stat, p)
    print(f"{model}: статистика={stat:.4f}, p-значение={p:.4f}")

# 2. Тест Левена для равенства дисперсий
levene_stat, levene_p = stats.levene(*mAP50_groups.values())
print("\nТест Левена (равенство дисперсий):")
print(f"Статистика={levene_stat:.4f}, p-значение={levene_p:.4f}")

# 3. Гистограммы для каждой модели, сохранение в отдельные файлы
for model, data in mAP50_groups.items():
    plt.figure(figsize=(4, 3))
    plt.hist(data, bins=10, alpha=0.7, color='blue')
    plt.title(model)
    plt.xlabel('mAP50')
    plt.ylabel('Частота')
    plt.grid(True)
    output_path = os.path.join(output_dir, f'hist_{model}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Сохранена гистограмма для {model} в {output_path}")