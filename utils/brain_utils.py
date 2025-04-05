from scipy.ndimage import binary_closing, binary_opening, label
from matplotlib import pyplot as plt
import numpy as np
import os


def _apply_window(data, center, width):
    """Применяет DICOM Window Level/Width"""
    min_val = center - width // 2
    max_val = center + width // 2
    data = np.clip(data, min_val, max_val)
    return data


def clear_dicom(
    dicom,
    pixels,
    filter_func=None,
    window:list[int, int]=None,
    window_center:int=None,
    window_width:int=None,
    bone_trh=150,
    water_trh=0
):
    """
    Скрипт очистки дикома, обязательно передавать окно формата (центр, ширина)
    """
    # Если все параметры пустые
    if not window and not window_center and not window_width:
        raise Exception("Должен быть передан хотя бы один из параметров window или window_center и window_width")
    
    if window:
        window_center, window_width = window
    
    if 'RescaleSlope' not in dicom and 'RescaleIntercept' not in dicom:
        raise Exception("В дикоме нет полей RescaleSlope и RescaleIntercept")
    
    # Если есть Rescale Slope/Intercept (для HU)
    pixels = pixels * dicom.RescaleSlope + dicom.RescaleIntercept
    
    # Фильтрация (удаление костей)
    if filter_func is not None:
        pixels = filter_func(pixels)
    # pixels[(pixels < water_trh) | (pixels > bone_trh)] = np.min(pixels)
    
    # Нормализация по окну (Window Level/Width)
    pixels = _apply_window(pixels, window_center, window_width)
    return pixels


def brain_split(
    data:np.ndarray,
    min_brain_trh:int=1,
    max_brain_trh:int=85,
    ):
    """Функция отделения мозга на изображении.
    
    Аргументы:
    - data (np.ndarray): Поступает data в формате очищенного от костей черепа (значения нормализованы).
    """
    # Маска для мозга
    brain_mask = (data > min_brain_trh) & (data < max_brain_trh)
    
    # Морфологические операции
    brain_mask = binary_closing(brain_mask, structure=np.ones((3, 3)))
    brain_mask = binary_opening(brain_mask, structure=np.ones((3, 3)))
    
    # Выбор наибольшей связной компоненты
    labeled_mask, num_labels = label(brain_mask)
    
    if num_labels == 0:
        return None
    
    largest_component = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
    final_brain_mask = (labeled_mask == largest_component)
    
    # Применение маски
    brain_only = data * final_brain_mask
    # Очистка всех что меньше 0 как 0
    brain_only = np.where(brain_only < 0, 0, brain_only)
    return brain_only