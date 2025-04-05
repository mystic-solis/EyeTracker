import os
import json
import shutil

import click


def flatten_png_files(source_dir):
    target_dir = source_dir
    
    # Рекурсивно ищем все .png файлы
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.png'):
                # Полный путь к исходному файлу
                src_file_path = os.path.join(root, file)
                # Новое имя файла в целевой папке (просто имя файла)
                dst_file_path = os.path.join(target_dir, file)

                # Если файл с таким именем уже существует, пропускаем
                if os.path.exists(dst_file_path):
                    continue

                # Копируем файл в целевую папку
                shutil.move(src_file_path, dst_file_path)
                print(f"Перемещен: {src_file_path} -> {dst_file_path}")

    # Удаляем все пустые папки после перемещения файлов
    remove_empty_folders(source_dir)


def remove_empty_folders(folder):
    # Рекурсивно удаляем пустые папки
    for root, dirs, files in os.walk(folder, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                # Пытаемся удалить папку, если она пуста
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    print(f"Удалена пустая папка: {dir_path}")
            except OSError as e:
                print(f"Ошибка при удалении папки {dir_path}: {e}")


def update_json_files(folder_path):
    # Проходим по всем файлам в папке
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # Читаем JSON-файл
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Обновляем поле file_name в массиве images
            for image in data.get('images', []):
                file_name = image.get('file_name', '')
                
                # Получаем номер случая и разделяем по '/'
                image['file_name'] = file_name.split('/')[-1]
            
            # Записываем обновленный JSON обратно в файл
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4, ensure_ascii=False)
            
            print(f"Обновлен файл: {filename}")


@click.command()
@click.option("--path", type=str, required=True, help="Путь к папке с датасетом.")
@click.option("--only-flatten", type=bool, default=False, help="Только извлечь данные из папок.")
def structure(path, only_flatten, **kwargs):
    """ Подготовка файловой структуры к конвертации датасета """
    
    annotations_path = os.path.join(path, 'annotations')
    images_path = os.path.join(path, 'images')
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Папка с COCO датасетом '{path}' не найдена.")
    
    flatten_png_files(images_path)
    print("Изображения подготовлены!")
    
    if not only_flatten:
        update_json_files(annotations_path)
        print("Json данные подготовлены!")


if __name__ == "__main__":
    structure()
