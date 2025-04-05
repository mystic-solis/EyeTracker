import json
import os
from pathlib import Path
import shutil
from click import Choice
import cv2
import click
import random

import numpy as np
from tqdm import tqdm
import copy

from utils.image_utils import get_blur_rotated_image, get_blured_image, get_rotated_image
from utils.utils import Image, polygon_area_and_bbox, process_directory, rotate_mask, rotate_point
import numpy as np


def add_image(flag, image_last_id, file_name, h, w, images_tmp):
    # Если флаг включен не сохраняем
    if flag is True:
        return image_last_id
    
    img = Image(id=image_last_id + 1, height=h, width=w, file_name=file_name)
    images_tmp.append(img)
    return image_last_id + 1, img.id


def get_coco_annotations(annotations_path_coco):
    annotations_jsons = process_directory(annotations_path_coco, find_sufix='json')
    
    if len(annotations_jsons) == 0:
        raise Exception("В папке аннотаций нет файлов .json. Укажите корневую папку COCO датасета")
    return annotations_jsons


def find_annotation_for_image(image_id, annotations):
    return list(filter(lambda annotation: annotation["image_id"] == image_id, annotations))


def get_max_ids(jsons_paths:list[str]):
    all_images, all_annotations = [], []
    
    for json_path in jsons_paths:
        with open(json_path, 'r') as j:
            json_data = json.loads(j.read())
        
        all_images.extend(json_data["images"])
        all_annotations.extend(json_data["annotations"])
    
    image_last_id = max(all_images, key=lambda x: x["id"])
    annotation_last_id = max(all_annotations, key=lambda x: x["id"])
    return image_last_id["id"]+1, annotation_last_id["id"]+1


def update_json(field, json_data, annot, annotation_last_id, img_blur_id, img_rotated_id, img_blur_rotated_id, annots_tmp, image, angle, height, width, no_blur, no_rotate):
    """
    Обновляет JSON-аннотацию, добавляя размытые и повернутые версии.
    """
    def add_annotation(annot_copy, img_id, ann_id):
        annot_copy["id"] = ann_id
        annot_copy["image_id"] = img_id
        annots_tmp.append(annot_copy)
        return ann_id + 1

    if not no_blur:
        annotation_last_id = add_annotation(copy.deepcopy(annot), img_blur_id, annotation_last_id + 1)

    if no_rotate:
        return annotation_last_id  # Если поворот не нужен, выходим сразу

    annot_rotated = copy.deepcopy(annot)

    if field == "seg":
        json_field_value = annot["segmentation"][0]
        rotated_segment = rotate_mask(json_field_value, image=image, angle=angle)
        
        area, bbox = polygon_area_and_bbox([rotated_segment], height, width)
        annot_rotated.update({"area": area.item(), "bbox": bbox, "segmentation": [rotated_segment]})
    
    elif field == "obb":
        current_angle = annot_rotated["attributes"]["rotation"]
        rotate_angle = (current_angle - angle) % 360
        annot_rotated["attributes"]["rotation"] = rotate_angle
        
        json_field_value = copy.copy(annot["bbox"])
        
        x1, y1, dx1, dy1 = json_field_value
    
        # Вычисляем все 4 угла прямоугольника
        theta_deg = np.radians(current_angle)
        center = (image.shape[1] // 2, image.shape[0] // 2)  # (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, theta_deg, 1.0)

        # Поворачиваем точку
        x_new, y_new = rotate_point(M, (x1, y1))
        
        M2 = cv2.getRotationMatrix2D((x_new + dx1, y_new + dy1), -theta_deg, 1.0)
        x_new, y_new = rotate_point(M2, (x_new, y_new))
        # NOTE Из-за особенностей работы CVAT.AI координаты слегка смещены относительно объекта. Угол и размеры рамки совпадают точно
        annot_rotated["bbox"] = [x_new, y_new, dx1, dy1]  # Тут должны быть координаты OBB
        
    annotation_last_id = add_annotation(annot_rotated, img_rotated_id, annotation_last_id + 1)

    if not no_blur:
        annotation_last_id = add_annotation(copy.deepcopy(annot_rotated), img_blur_rotated_id, annotation_last_id + 1)

    json_data["annotations"].extend(annots_tmp)
    return annotation_last_id


@click.command()
@click.option('--path', required=True, type=str, help="Папка с изображениями для аугментации.")
@click.option('--annot-type', type=Choice(['seg', 'obb']), default='seg', help="С какими объектами работать (сегментация или ориентированные прямоугольники).")
@click.option('--num', default=1, type=int, help="Кол-во раз аугментации.")
@click.option('--output', default="augmented-dataset", type=str, help="Папка для сохранения результатов.")
@click.option('--blur', default=5, type=int, help="Размерность ядра блюра (1-почти нет блюра, 9-сильный блюр).")
@click.option('--angle', default=None, type=int, help="Угол поворота изображения.")
@click.option('--no-blur', is_flag=True, help="Не использовать блюр.")
@click.option('--no-rotate', is_flag=True, help="Не использовать поворот.")
@click.option('--clear-out-dir', is_flag=True, help="Не очищать перед работой выходную директорию.")
def augment(path, annot_type, num, angle, blur, no_blur, no_rotate, output, clear_out_dir):
    """ Аугментация датасета """
    
    # Проверка
    if not os.path.exists(path):
        raise Exception("Папка с изображениями не найдена!")
    
    if clear_out_dir:
        shutil.rmtree(output, ignore_errors=True)
    
    # Высчитывание аннотаций и изображений
    annotations_path_coco = os.path.join(path, "annotations")
    images_path_coco = os.path.join(path, "images")
    
    annotations_path = os.path.join(output, "annotations")
    images_path = os.path.join(output, "images")
    
    os.makedirs(output, exist_ok=True)
    os.makedirs(annotations_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    
    # Получение всех COCO аннотаций
    annotations_jsons = get_coco_annotations(annotations_path_coco)
    
    # Получение максимальных значений id для изображений и аннотаций для безошибочной записи
    image_last_id, annotation_last_id = get_max_ids(annotations_jsons)
    
    for json_path in tqdm(annotations_jsons):
        json_name = Path(json_path).name
        
        with open(json_path, 'r') as j:
            json_data = json.loads(j.read())
        
        json_images = copy.deepcopy(json_data["images"])
        json_annotations = copy.deepcopy(json_data["annotations"])
        
        for image_dict in tqdm(json_images):
            image_file_path = os.path.join(images_path_coco, image_dict["file_name"])
            
            for aug_index in range(num):
                name = Path(image_file_path).stem
                suffix = Path(image_file_path).suffix
                
                # Имена аугментированных изображений
                blur_file_name = f"{name}-blur_a{aug_index}{suffix}"
                rotated_file_name = f"{name}-rotate_a{aug_index}{suffix}"
                blur_rotated_file_name = f"{name}-blur-rotate_a{aug_index}{suffix}"
                
                image = cv2.imread(image_file_path)
                
                # Сохранение в директорию исходного файла
                original_output_path = os.path.join(images_path, f"{name}{suffix}")
                cv2.imwrite(original_output_path, image)
                
                # Блюр изображения
                blurred_image = get_blured_image(blur, images_path, blur_file_name, image, no_blur)
                
                # Поворот оригинала случайный угол от -90 до 90 градусов
                angle = angle or random.uniform(-90, 90)
                
                rotated_image = get_rotated_image(image, rotated_file_name, images_path, angle, no_rotate)
                
                # Поворот размытого изображения
                rotated_blurred_image = get_blur_rotated_image(images_path, angle, blur_rotated_file_name, no_blur, no_rotate)
                
                # Сохранение всех изображений в JSON
                h, w = image.shape[:2]
                # Сохраняем в JSON только те у которых изменения. Исходное изображение мы просто скопировали в выходную директорию
                images_tmp = []
                
                image_last_id, img_blur_id = add_image(no_blur, image_last_id, blur_file_name, h, w, images_tmp)
                image_last_id, img_rotated_id = add_image(no_rotate, image_last_id, rotated_file_name, h, w, images_tmp)
                image_last_id, img_blur_rotated_id = add_image(no_blur and no_rotate, image_last_id, blur_rotated_file_name, h, w, images_tmp)
                
                images_tmp = [i.model_dump() for i in images_tmp]
                json_data["images"].extend(images_tmp)
                
                # Работа с аннотациями
                image_annots = find_annotation_for_image(image_dict["id"], json_annotations)
                
                for annot in image_annots:
                    annots_tmp = []
                    
                    annotation_last_id = update_json(
                        annot_type,
                        json_data,
                        annot,
                        annotation_last_id,
                        img_blur_id,
                        img_rotated_id,
                        img_blur_rotated_id,
                        annots_tmp,
                        image,
                        angle,
                        h, w,
                        no_blur,
                        no_rotate
                    )
        # Работа с JSON
        json_new_path = os.path.join(annotations_path, json_name)
        with open(json_new_path, 'w') as j:
            json.dump(json_data, j)


if __name__ == "__main__":
    augment()