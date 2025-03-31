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


def process_directory(input_dir, log):
    """
    Обрабатывает директорию, проверяет её существование и возвращает список файлов .dcm.
    
    :param input_dir: Путь к директории.
    :return: Список файлов .dcm.
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
    dicom_files = glob(os.path.join(input_dir, "**", "*.dcm"), recursive=True)
    
    # Если файлов нет, выводим предупреждение
    if not dicom_files:
        log.error(f"В директории '{input_dir}' не найдено файлов .dcm.")
    
    return dicom_files


def convert_to_png(data, name:str=None, output_dir:str=None, save=True, verbose=False):
    """
    Сохраняет массив NumPy в формате PNG.
    
    :param array: Массив NumPy (2D или 3D).
    :param name: Имя файла для сохранения.
    """
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

    # Сохранение изображения
    if save and output_dir and name:
        os.makedirs(output_dir, exist_ok=True)
        image.save(f"{output_dir}/{name}")
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