# YOLO FLOPS Calculator

Подсчет FLOPS для моделей YOLO с обычным инферансом и SAHI.

## Установка

```bash
git clone https://github.com/TomilovR/improved-doodle.git
cd .\improved-doodle\
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
Скрипт автоматически использует обученную модель `best.pt` если доступна, иначе `yolov8s.pt`. 

Также вычисляется mAP@0.5 на 5 тестовых изображениях с аннотациями из файла `sample_images/_annotations.csv`.

## Файлы

- `flops_calculator.py` - основной скрипт
- `requirements.txt` - зависимости
- `sample_images/` - тестовые изображения с аннотациями
- `yolov8s.pt` - предобученная модель
- `best.pt` - обученная модель
