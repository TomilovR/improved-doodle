# YOLO FLOPS Calculator

Подсчет FLOPS для моделей YOLO с обычным инферансом и SAHI.

## Установка

```bash
git clone https://github.com/your-username/yolo-flops-calculator.git
cd yolo-flops-calculator
pip install -r requirements.txt
```

## Запуск

```bash
python flops_calculator.py
```

## Результат

```
YOLO FLOPS Calculator
Normal YOLO FLOPS: 3.64e+10
SAHI FLOPS: 7.28e+11
SAHI increase: 20.0x
mAP@0.5: 0.847
```

## Конфигурация

Параметры в `flops_calculator.py`:

```python
SLICE_HEIGHT = 1024
SLICE_WIDTH = 1024
OVERLAP_HEIGHT_RATIO = 0.25
OVERLAP_WIDTH_RATIO = 0.25
IMAGE_SIZE = 1024
CONF_THRESHOLD = 0.4
```

Скрипт автоматически использует обученную модель `best.pt` если доступна, иначе `yolov8s.pt`. 

Также вычисляется mAP@0.5 на 5 тестовых изображениях с аннотациями из файла `sample_images/_annotations.csv`.

## Файлы

- `flops_calculator.py` - основной скрипт
- `requirements.txt` - зависимости
- `sample_images/` - тестовые изображения с аннотациями
- `yolov8s.pt` - предобученная модель
- `best.pt` - обученная модель
