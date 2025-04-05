import cv2
from pycocotools import mask as mask_util
from glob import glob
from PIL import Image as PILImage
import numpy as np 
import os

from pydantic import BaseModel, Field


class Image(BaseModel):
    id: int
    width: int
    height: int
    file_name: str
    license: int = 0
    flickr_url: str = ""
    coco_url: str = ""
    date_captured: int = 0


class Annotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    segmentation: list
    area: int
    bbox: list
    iscrowd: int = 0
    attributes: dict = Field(default={"occluded": False})


def get_background_color(image):
    # Берём цвет из верхнего левого угла
    return tuple(map(int, image[0, 0]))


def rotate_image(image, angle, bg_color):
    if not bg_color:
        bg_color = get_background_color(image)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
    return rotated


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


def rotate_point(M, point):
    """Применяет матрицу поворота к точке"""
    x, y = point
    homogeneous_point = np.array([x, y, 1])  # Преобразуем в однородные координаты
    new_point = np.dot(M, homogeneous_point)  # Умножаем на матрицу поворота
    return tuple(new_point[:2])  # Возвращаем (x', y')


def split_by_proportions(arr, proportions):
    """
    Разделяет массив на части по заданным пропорциям.

    :param arr: Входной массив (например, np.array или список).
    :param proportions: Список пропорций (например, [0.7, 0.2, 0.1]).
    :return: Список массивов, разделенных по пропорциям.
    """
    # Вычисляем размеры каждой части
    total_size = len(arr)
    sizes = [int(total_size * prop) for prop in proportions]
    sizes[-1] = total_size - sum(sizes[:-1])  # Корректируем последнюю часть

    # Разделяем массив
    split_indices = np.cumsum(sizes)[:-1]  # Индексы для разделения
    return np.split(arr, split_indices)


def get_intermediate_folder(path: str, input_dir: str, default: str = "root"):
    # input_dir/ВОТ ЭТА ЧАСТЬ НАМ НУЖНА/имя файла.dcm
    # model/V1/d2/b5//IMG-0001-00125.dcm
    # model/IMG-0001-00125.dcm
    # Отделяем известный корень
    relative_part = os.path.relpath(path, start=input_dir)
    # some/folder/file.dcm
    # file.dcm
    split_symbol = '/' if "/" in relative_part else '\\'
    parts = relative_part.split(split_symbol)
    if len(parts) > 1:  # Если есть хотя бы одна папка
        return parts[0]
    return default

def count_all_numbers(paths, input_dir, default_folder):
    letter_counts = {}
    
    # Перебираем пути и считаем
    for path in paths:
        folder = get_intermediate_folder(path, input_dir, default_folder)
        
        if folder not in letter_counts:
            letter_counts[folder] = 0
        letter_counts[folder] += 1
    return letter_counts


def process_directory(input_dir, find_sufix:str="png"):
    """
    Обрабатывает директорию, проверяет её существование и возвращает список файлов.
    
    :param input_dir: Путь к директории.
    :return: Список файлов.
    """
    # Нормализация пути (убираем лишние символы, приводим к абсолютному пути)
    input_dir = os.path.normpath(input_dir)
    
    # Проверяем, существует ли путь
    if not os.path.exists(input_dir):
        raise ValueError(f"Путь '{input_dir}' не существует!")
    
    # Проверяем, что это директория
    if not os.path.isdir(input_dir):
        raise ValueError(f"'{input_dir}' не является директорией!")
    
    # Ищем все файлы .dcm в директории
    files = glob(os.path.join(input_dir, "**", f"*.{find_sufix}"), recursive=True)
    
    # Если файлов нет, выводим предупреждение
    if not files:
        print(f"В директории '{input_dir}' не найдено файлов расширения {find_sufix}.")
    print(f"Файлов {find_sufix} найдено: {len(files)}")
    return files


def save_to_png(data, name:str=None, output_dir:str=None, save=True, angle:int=None, verbose=False):
    """
    Сохраняет массив NumPy в формате PNG.
    
    :param array: Массив NumPy (2D или 3D).
    :param name: Имя файла для сохранения.
    """
    if data is None:
        return None
    
    # Нормализация массива в диапазон 0–255
    if data.dtype != np.uint8:
        array_normalized = ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8)
    else:
        array_normalized = data

    # Преобразование массива в изображение
    if array_normalized.ndim == 2:  # Grayscale (2D)
        image = PILImage.fromarray(array_normalized, mode='L')
    elif array_normalized.ndim == 3:  # RGB (3D)
        image = PILImage.fromarray(array_normalized, mode='RGB')
    else:
        raise ValueError("Массив должен быть 2D (grayscale) или 3D (RGB).")
    
    # Если указан угол то поворачиваем
    if angle:
        image = rotate_image(image=image, angle=angle)
    
    # Сохранение изображения
    if save and output_dir and name:
        os.makedirs(output_dir, exist_ok=True)
        image.save(os.path.join(output_dir, name))
    if verbose:
        print(f"Изображение сохранено как {name}")
    return image


def normalize(data, new_min=-1000, new_max=1000):
    """Нормализация через линейное преобразование.
    Нормализованные данные = (данные - min) / (max - min) * (new_max - new_min) + new_min
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * (new_max - new_min) + new_min


def pixel_to_hu(intercept:float, slope:float, dicom=None, pixel_data=None, norm=True, new_min=-1000, new_max=1000):
    """Преобразование пикселей в HU.
    """
    if pixel_data is None:
        if dicom is None: raise ValueError("Необходимо передать pixel_data или dicom")
        pixel_data = dicom.pixel_array
    
    data = pixel_data * slope + intercept
    
    if norm:
        return normalize(data, new_min, new_max)
    return data


def save(dicom, data:np.ndarray, path:str, pixel_data=None):
    """Корректное сохранение DICOM файла.
    """
    if pixel_data is None:
        dicom.PixelData = data.tobytes()
    else:
        dicom.PixelData = pixel_data
    dicom.Rows, dicom.Columns = data.shape
    # Сохранение
    dicom.save_as(path)


def convert_hu_to_pixel(*args, pixel_min, pixel_max, hu_min=-1000, hu_max=1000):
    """Перевод из HU (с границами -1000 и 1000) в Pixel с его границами (-2000 и 2800)"""
    modifiers = np.array([*args, hu_min, hu_max])
    modifiers = normalize(data=modifiers, new_min=pixel_min, new_max=pixel_max)
    return modifiers[:-2]


def polygon_area_and_bbox(polygon, height, width):
    """Calculate area of object's polygon and bounding box around it
    Args:
        polygon: objects contour represented as 2D array
        height: height of object's region (use full image)
        width: width of object's region (use full image)
    """
    rle = mask_util.frPyObjects(polygon, height, width)
    area = mask_util.area(rle)
    bbox = mask_util.toBbox(rle)
    bbox = [min(bbox[:, 0]),
            min(bbox[:, 1]),
            max(bbox[:, 0] + bbox[:, 2]) - min(bbox[:, 0]),
            max(bbox[:, 1] + bbox[:, 3]) - min(bbox[:, 1])]
    return area, bbox


if __name__ == "__main__":
    data = np.array([[100, 200, 500]]) 
    
    normalized = normalize(data)
    
    print(f"{data = }")
    print(f"{normalized = }")