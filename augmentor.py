from glob import glob
import json
import os
from pathlib import Path
import shutil
import cv2
import click
import random

import numpy as np
from tqdm import tqdm
import copy

from utils.utils import Image, polygon_area_and_bbox


def rotate_mask(points:list[int], image, angle):
    # Поворачиваем точки сегментации
    points = np.array(points).reshape(-1, 2)
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated_points = []
    for x, y in points:
        new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        rotated_points.append((new_x, new_y))
    return np.array(rotated_points).flatten().tolist()


def find_annotation_for_image(image_id, annotations):
    return list(filter(lambda annotation: annotation["image_id"] == image_id, annotations))


def apply_blur(image, core=None):
    if not core: core = (3, 3)
    return cv2.GaussianBlur(image, core, 0)


def rotate_image(image, angle, bg_color):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
    return rotated


def get_background_color(image):
    # Берём цвет из верхнего левого угла
    return tuple(map(int, image[0, 0]))


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


@click.command()
@click.option('--path', required=True, type=str, help="Папка с изображениями для аугментации.")
@click.option('--output', default="augmented-dataset", type=str, help="Папка для сохранения результатов.")
@click.option('--blur', default=5, type=int, help="Разменрность ядра блюра (1-почти нет блюра, 9-сильный блюр).")
@click.option('--no-blur', default=False, type=int, help="Не использовать блюр.")
@click.option('--no-rotate', default=False, type=int, help="Не использовать поворот.")
@click.option('--clear-out-dir', is_flag=True, help="Не очищать перед работой выходную директорию.")
def main(**kwargs):
    path = kwargs.get("path")
    blur = kwargs.get("blur")
    no_blur = kwargs.get("no_blur")
    no_rotate = kwargs.get("no_rotate")
    output_dir = kwargs.get("output")
    clear_out_dir = kwargs.get("clear_out_dir")
    
    if not os.path.exists(path):
        raise Exception("Папка с изображениями не найдена!")
    
    if clear_out_dir:
        shutil.rmtree(output_dir, ignore_errors=True)
    
    # Высчитывание аннотаций и изображений
    annotations_path_coco = os.path.join(path, "annotations")
    images_path_coco = os.path.join(path, "images")
    
    annotations_path = os.path.join(output_dir, "annotations")
    images_path = os.path.join(output_dir, "images")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(annotations_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    
    annotations_jsons = glob(os.path.join(annotations_path_coco, "**", "*.json"), recursive=True)
    if len(annotations_jsons) == 0:
        raise Exception("В папке аннотаций нет файлов .json. Укажите корневую папку COCO датасета")
    
    # Получение максимальных значений id для изображений и аннотаций для безошибочной записи
    image_last_id, annotation_last_id = get_max_ids(annotations_jsons)
    
    for json_path in tqdm(annotations_jsons):
        json_name = Path(json_path).name
        
        with open(json_path, 'r') as j:
            json_data = json.loads(j.read())
        
        json_images = copy.deepcopy(json_data["images"])
        json_annotations = copy.deepcopy(json_data["annotations"])
        
        for image_dict in tqdm(json_images):
            file = os.path.join(images_path_coco, image_dict["file_name"])
            
            name = Path(file).stem
            suffix = Path(file).suffix
            
            image = cv2.imread(file)
            
            # Цвет фона определяется по верхнему левому углу
            bg_color = get_background_color(image)
            
            # Сохранение в директорию исходного файла
            original_output_path = os.path.join(images_path, f"{name}{suffix}")
            cv2.imwrite(original_output_path, image)
            
            # Блюр изображения
            if not no_blur:
                blur_file_name = f"{name}-blur{suffix}"
                blurred = apply_blur(image, core=(blur, blur))
                blur_output_path = os.path.join(images_path, blur_file_name)
                cv2.imwrite(blur_output_path, blurred)
            
            
            # Поворот оригинала
            if not no_rotate:
                rotated_file_name = f"{name}-rotate{suffix}"
                angle = random.uniform(0, 360)  # Случайный угол от 0 до 360 градусов
                rotated = rotate_image(image, angle, bg_color)
                rotated_output_path = os.path.join(images_path, rotated_file_name)
                cv2.imwrite(rotated_output_path, rotated)
            
                # Поворот этого же размытого изображения
                blur_rotated_file_name = f"{name}-blur-rotate{suffix}"
                rotated_blurred = rotate_image(blurred, angle, bg_color)
                rotated_blur_output_path = os.path.join(images_path, blur_rotated_file_name)
                cv2.imwrite(rotated_blur_output_path, rotated_blurred)
            
            # Сохранение всех изображений в JSON
            h, w = image.shape[:2]
            # Сохраняем в JSON только те у которых изменения. Исходное изображение мы просто скопировали в выходную директорию
            images_tmp = []
            if not no_blur:
                img_blur = Image(id=image_last_id + 1, height=h, width=w, file_name=blur_file_name)
                images_tmp.append(img_blur)
                image_last_id += 1
            
            if not no_rotate:
                img_rotated = Image(id=image_last_id + 1, height=h, width=w, file_name=rotated_file_name)
                img_blur_rotated = Image(id=image_last_id + 2, height=h, width=w, file_name=blur_rotated_file_name)
                images_tmp.append(img_rotated)
                images_tmp.append(img_blur_rotated)
                image_last_id += 2
            
            images_tmp = [i.model_dump() for i in images_tmp]
            json_data["images"].extend(images_tmp)
            
            # для блюра только id меняем у копии словаря
            # для поворота маску и id у копии словаря
            
            # Работа с аннотациями
            image_annots = find_annotation_for_image(image_dict["id"], json_annotations)
            
            for annot in image_annots:
                annots_tmp = []
                
                if not no_blur:
                    annot_blur = copy.deepcopy(annot)
                    # Блюр
                    annot_blur["id"] = annotation_last_id + 1
                    annot_blur["image_id"] = img_blur.id
                    annots_tmp.append(annot_blur)
                    annotation_last_id += 1
                
                # Поворот
                if not no_rotate:
                    annot_rotated = copy.deepcopy(annot)
                    rotated_segment = rotate_mask(annot["segmentation"][0], image=image, angle=angle)
                    rotated_segment = np.array([rotated_segment])
                    area, bbox = polygon_area_and_bbox(rotated_segment, h, w)
                    annot_rotated["id"] = annotation_last_id + 1
                    annot_rotated["image_id"] = img_rotated.id
                    annot_rotated["segmentation"] = rotated_segment.tolist()
                    annot_rotated["area"] = area.item()
                    annot_rotated["bbox"] = bbox
                
                    # Поворот + блюр (копируем Повернутую аннотацию и меняем id)
                    annot_blur_rotated = copy.deepcopy(annot_rotated)
                    annot_blur_rotated["id"] = annotation_last_id + 2
                    annot_blur_rotated["image_id"] = img_blur_rotated.id
                    
                    annots_tmp.append(annot_rotated)
                    annots_tmp.append(annot_blur_rotated)
                    json_data["annotations"].extend(annots_tmp)
                    annotation_last_id += 2
        
        # Работа с JSON
        json_new_path = os.path.join(annotations_path, json_name)
        with open(json_new_path, 'w') as j:
            json.dump(json_data, j)


if __name__ == "__main__":
    # main("--path coco-dataset".split())
    main()
