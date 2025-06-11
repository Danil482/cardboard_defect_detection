import os

# Путь к папке с изображениями
folder_path = 'unlabled_images'

# Получаем список всех файлов в папке
files = os.listdir(folder_path)

# Фильтруем только изображения (можно добавить другие форматы, если нужно)
image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]

# Начальное значение для именования
start_number = 1057

# Переименование файлов
for i, filename in enumerate(images):
    # Создаем новое имя файла
    new_name = f"{start_number + i}.jpg"

    # Полные пути к старому и новому файлам
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_name)

    # Переименовываем файл
    os.rename(old_file, new_file)

print("Переименование завершено.")
