#  COCO to YOLO converter for instance segmentation (YOLOv8-seg) and oriented bounding box detection (YOLOv8-obb)

# Russian Version of README:

Репозиторий позволяет преобразовать разметку формата COCO в формат, поддерживаемый для обучения моделей YOLOv8-seg (инстанс сегментация) и YOLOv8-obb (детекция повернутых боксов).

Ключевое применение репозитория -> работа с выгруженной разметкой **полигонов** (или **повернутых прямоугольников** в случае с YOLOv8-obb) из приложения CVAT в формате COCO 1.0 (с указанием режима save images = True).

Если же используете без CVAT, то убедитесь перед запуском, что ваша папка с COCO датасетом имеет такую структуру:
```
COCO_dataset/
|-- annotations/
|   |-- instances_train.json
|   |-- instances_val.json
|-- images/
|   |-- image1.jpg
|   |-- image2.jpg
|   |-- ...
```
PS: Для задчи инстанс сегментации имеется также поддержка моделей Ultralytics YOLO12-seg, YOLO11-seg, YOLOv9-seg и YOLOv5-seg и других (так как у них аналогичная разметка с версией v8)

## Примеры использования:

Пример использования репозитория для задачи ***YOLOv8-seg*** представлен в видео на YouTube - [__ССЫЛКА__](https://www.youtube.com/watch?v=FF3mIWF0vFs&t=6s?t=34m49s) <br/>
Пример использования репозитория для задачи ***YOLOv8-obb*** представлен в видео на YouTube - [__ССЫЛКА__](https://www.youtube.com/watch?v=CZ_kZlto3IY&t=920s?t=20m2s)

## Установка:
```
git clone https://github.com/Koldim2001/COCO_to_YOLOv8.git

cd COCO_to_YOLOv8

pip install -r requirements.txt
```

## Как запускать код:

__Классический подход c предустановленным в CVAT разделением на train/val/test (у тасок определен Subset):__
```
python coco_to_yolo.py --coco_dataset="dataset_folder" --lang_ru=True
```
__Вариант с авторазделением на train и val:__

```
python coco_to_yolo.py --coco_dataset="dataset_folder" --autosplit=True --percent_val=30 --lang_ru=True
```

Список параметров с пояснениями, которые можно передать на вход программы перед ее запуском в cli:
```bash
  --coco_dataset TEXT   Папка с датасетом формата COCO 1.0 (можно выгрузить из
                        CVAT). По умолчанию "COCO_dataset"

  --yolo_dataset TEXT   Папка с итоговым датасетом формата YOLOv8. По
                        умолчанию "YOLO_dataset"

  --print_info BOOLEAN  Вкл/Выкл режима вывода логов обработки. По умолчанию
                        отключен

  --autosplit BOOLEAN   Вкл/Выкл режима автоматического разделения на
                        train/val. По умолчанию отключен (берет согласно
                        разметке CVAT)

  --percent_val FLOAT   Процент данных на валидацию при выборе режима
                        autosplit=True. По умолчанию 25%

  --lang_ru BOOLEAN     Устанавливает русский язык комментариев, если выбрано 
                        значение True. По умолчанию английский 

  --help                Покажет существующие варианты парсинга аргументов в CLI
  ```
