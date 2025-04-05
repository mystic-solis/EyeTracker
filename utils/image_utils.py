import os

import cv2
from utils.utils import get_background_color, rotate_image


def apply_blur(image, core=None):
    if not core: core = (3, 3)
    return cv2.GaussianBlur(image, core, 0)


def get_blur_rotated_image(images_path, image, angle, blur, file_name, no_blur, no_rotate, save=True):
    rotated_blur_output_path = os.path.join(images_path, file_name)
    
    blurred_image = get_blured_image(blur, images_path, '', image, no_blur, save=False)
    rotated_blurred_image = get_rotated_image(blurred_image, '', images_path, angle, no_rotate, save=False)
    
    # Если флаги не включены то сохраняем изображение
    if not no_blur and not no_rotate and save:
        cv2.imwrite(rotated_blur_output_path, rotated_blurred_image)
    return rotated_blurred_image


def get_rotated_image(image, file_name, images_path, angle, no_rotate, save=True):
    # Цвет фона определяется по верхнему левому углу
    bg_color = get_background_color(image)
        
    rotated_image = rotate_image(image, angle, bg_color)
    rotated_output_path = os.path.join(images_path, file_name)
    
    # Если поворот разрешен (не включен как флаг), то сохраняем
    if not no_rotate and save:
        cv2.imwrite(rotated_output_path, rotated_image)
    return rotated_image


def get_blured_image(blur, images_path, file_name, image, no_blur, save=True):
    blur_output_path = os.path.join(images_path, file_name)
    blurred_image = apply_blur(image, core=(blur, blur))
    
    # Если блюр разрешен (не включен как флаг), то сохраняем
    if not no_blur and save:
        cv2.imwrite(blur_output_path, blurred_image)
    return blurred_image