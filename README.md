# CrowdTracking

## Описание проекта
CrowdTracking — это Python-приложение для детекции людей на видео с помощью предобученной модели YOLO. Программа обрабатывает входное видео, находит всех людей на каждом кадре и сохраняет новый видеоролик с отрисованными рамками, именами классов и значениями уверенности для каждого обнаруженного человека.

## Структура проекта
- **main.py** — Точка входа. Управляет обработкой видео, детекцией и сохранением результата.
- **modules/yolo_model.py** — Класс-обёртка для загрузки и инференса модели YOLO, выбора устройства (CPU/GPU/MPS).
- **modules/video_stream.py** — Класс для работы с видеопотоком: открытие, чтение кадров, освобождение ресурсов.
- **modules/utils.py** — Парсинг аргументов командной строки и вспомогательные функции.
- **crowd.mp4** — Пример входного видео.
- **output.mp4** — Пример выходного видео с детекцией.
- **requirements.txt** — Список зависимостей для установки.

## Возможности
- Детекция людей на видео с помощью моделей YOLO (You Only Look Once).
- На выходе — видео с рамками, подписями класса и уровнем уверенности для каждого человека.
- Прогресс-бар для отслеживания статуса обработки.
- Опциональный вывод результата в реальном времени на экран.
- Кроссплатформенность: работает на Linux, macOS и Windows.

## Установка
1. Клонируйте репозиторий или скачайте исходный код:
   ```bash
   git clone https://github.com/mr1necs/CrowdTracking.git
   cd CrowdTracking
   ```
2. Установите необходимые зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Использование
Запустите программу из командной строки:
```bash
python main.py --input crowd.mp4 --output output.mp4
```

### Аргументы командной строки
- `-i`, `--input`   : Путь к входному видеофайлу (обязательный параметр).
- `-o`, `--output`  : Путь для сохранения выходного видео (по умолчанию: `output.mp4`).
- `-m`, `--model`   : Путь к файлу модели YOLO или имя модели (по умолчанию: `yolo11n.pt`).
- `-d`, `--device`  : Устройство для инференса: `cpu`, `cuda` или `mps` (по умолчанию: `cpu`).
- `-s`, `--show`    : Показывать видео с детекцией в окне во время обработки (по умолчанию: выкл.).

### Пример
```bash
python main.py --input crowd.mp4 --output output.mp4 --model yolo11n.pt --device mps --show
```

## Результат
- На выходе получается видеофайл (например, `output.mp4`) с отрисованными рамками вокруг каждого обнаруженного человека.
- Для каждой рамки указывается имя класса (`Person`) и уровень уверенности.
<div align="center">
  <img src="data/before.gif" width="45%" alt="До обработки"/>
  <img src="data/after.gif" width="45%" alt="После обработки"/>
</div>

## Кроссплатформенность
Программа полностью кроссплатформенная и одинаково работает на Linux, macOS и Windows. Все зависимости устанавливаются через PyPI и не требуют специальных шагов для разных платформ.

## Примечания
- По умолчанию используется модель YOLO `yolo11n.pt`, но вы можете указать любую совместимую модель YOLO через аргумент `--model`.
- Для наилучших результатов используйте качественные исходные видео и убедитесь, что нужные веса модели доступны.
- Программа использует только open-source библиотеки и не требует проприетарного ПО.

## Лицензия
Проект распространяется под лицензией MIT.

## Выводы и рекомендации

### Анализ качества распознавания
- Программа успешно выполняет детекцию людей на видео, корректно отрисовывает рамки, имена классов и уровень уверенности для каждого объекта.
- В большинстве случаев модель YOLOv11 демонстрирует высокую точность и скорость работы на стандартных видеороликах с хорошим освещением и разрешением.
- В сложных условиях (перекрытия, низкое освещение, малая видимость объектов) возможны пропуски или ложные срабатывания, что типично для универсальных моделей.
- Визуализация реализована так, чтобы не мешать просмотру исходного видео: рамки и подписи не перекрывают важные детали.

### Пути дальнейшего улучшения
- Использовать более крупные или специализированные модели YOLO (например, yolov11m, yolov11l) для повышения точности на сложных сценах.
- Дообучить модель на собственных данных, если в видео встречаются специфические ракурсы, одежда или условия съёмки.
- Добавить постобработку для фильтрации ложных срабатываний (например, по минимальной уверенности или размеру объекта).
- Реализовать трекинг объектов между кадрами для более стабильного отображения и анализа перемещений.
- Добавить автоматическую оценку качества работы (например, подсчёт количества пропущенных/ложных детекций при наличии разметки).