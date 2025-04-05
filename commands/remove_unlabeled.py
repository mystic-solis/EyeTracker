import json
import click
import os
import random

from utils.utils import process_directory


def find_unannotated_images(data, verbose):
    """Function which find not annotated images and return list of them"""
    
    annotated_images = [annot["image_id"] for annot in data["annotations"]]
    images = [img for img in data["images"]]
    
    ff = lambda x: x.get("id") not in annotated_images
    non_annot_images = list(filter(ff, images))
    
    images_len = len(images)
    annot_len = len(annotated_images)
    non_annot_len = len(non_annot_images)
    
    if verbose:
        print(f"- Найдено изображений: {images_len}")
        print(f"- Найдено аннотаций: {annot_len}")
        print(f"- Не аннотировано: {non_annot_len}")
    return annot_len, non_annot_len, non_annot_images


def get_images_to_delete(images, num):
    to_delete = []
    for _ in range(num):
        choice = random.choice(images)
        to_delete.append(choice)
        images.remove(choice)
    return to_delete


def delete_images(image_paths, images_dir, simulate):
    for idx, image in enumerate(image_paths):
        path = os.path.join(images_dir, image)
        if simulate: continue
        if os.path.isfile(path):
            os.remove(path)


@click.command()
@click.option("-p", "--path", required=True, type=str, help='Путь к папке COCO датасета.')
@click.option("-v", "--verbose", default=False, type=bool, help='Выводить ли текст.')
@click.option("-s", "--simulation", default=False, type=bool, help='Симуляция работы скрипта.')
@click.option("--mode", default="1:1", type=str, help='Коэффициент аннотированных и нет изображений. 1:1, 1:0.5, 1:2 и так далее')
def remove_unlabeled(path, verbose, simulation, mode, **kwargs):
    """ Удаление неразмеченных данных """
    
    # Проверка существования датасета
    if not os.path.exists(path):
        raise Exception("Папка COCO датасета не найдена!")
    
    # Высчитывание аннотаций и изображений
    annotations_path = os.path.join(path, "annotations")
    images_path = os.path.join(path, "images")
    
    # Поиск всех json в пути
    annotations_jsons = process_directory(annotations_path, find_sufix='json')
    
    # Обработка всех json файлов
    for json_path in annotations_jsons:
        print(f"\nФайл: {json_path}")
        
        with open(json_path, 'r') as j:
            data = json.loads(j.read())
            
        annot_len, non_annot_len, non_annot_paths = find_unannotated_images(data, verbose=verbose)
        
        # Вычисляем коэффициент удаления изображений
        num_to_delete = int(non_annot_len - annot_len*mode)
        if verbose:
            print(f"Будет удалено: {num_to_delete} изображений. Останется: {non_annot_len - num_to_delete} не аннотированных")
        
        # Рандомно выбираем изображения для удаления
        to_delete = get_images_to_delete(images=non_annot_paths, num=num_to_delete)
        
        # Убираем удаляемые изображения из общего списка
        for i in to_delete:
            data["images"].remove(i)
        
        # Удяляем изображения
        to_delete = [i["file_name"] for i in to_delete]
        delete_images(to_delete, images_dir=images_path, simulate=simulation)
        
        # Обновляем файл json
        if not simulation:
            with open(json_path, 'w') as j:
                json.dump(data, j)
        
        print(f"Файл обработан!")


if __name__ == "__main__":
    # main("-p test --mode 1:1 -s 1 -v 1".split())
    remove_unlabeled()