"""Локализация веб-интерфейса (RU по умолчанию / EN).

Все видимые строки идут через t(key). Язык хранится в st.session_state["lang"]
и переключается селектором в сайдбаре - при смене Streamlit перерисовывает весь
интерфейс, поэтому переводятся и пункты комбобоксов (через format_func -> t),
и подсказки help=, и подписи.
"""
import streamlit as st

LANGUAGES = {"ru": "Русский", "en": "English"}
DEFAULT_LANG = "ru"

TRANSLATIONS = {
    "ru": {
        # - общее / сайдбар -
        "app_title": "Верификатор синтетических данных - сегментация дефектов",
        "app_caption": "Обучение U-Net на смеси реальных и синтетических данных",
        "language_label": "Язык интерфейса",
        "nav_help": "Справка",
        "nav_train": "Обучение",
        "nav_contours": "Контуры",
        "nav_results": "Результаты",
        # - обучение -
        "train_header": "Обучение и верификация",
        "real_path_label": "Папка реальных данных",
        "real_path_help": "Путь к датасету с подпапками images/ и bitmaps/ (реальные снимки).",
        "synth_path_label": "Папка синтетических данных",
        "synth_path_help": "Путь к датасету синтетики (та же структура images/ + bitmaps/).",
        "mode_label": "Режим",
        "mode_help": "sweep - перебор доли синтетики 0...100%; single - один прогон с заданными долями.",
        "mode_sweep": "Перебор доли синтетики (sweep)",
        "mode_single": "Один прогон (single)",
        "epochs_label": "Количество эпох",
        "epochs_help": "Сколько раз модель проходит обучающую выборку за один прогон.",
        "batch_label": "Размер батча",
        "batch_help": "Сколько изображений обрабатывается за один шаг обучения.",
        "test_size_label": "Размер тестовой выборки",
        "test_size_help": "Сколько реальных изображений отделить под тест (не пересекается с обучением).",
        "max_images_label": "Лимит изображений (0 = без лимита)",
        "max_images_help": "Обрезать датасеты до N изображений - для быстрых демо-прогонов на CPU.",
        "real_size_label": "Доля реальных данных",
        "real_size_help": "Только для режима single: какую долю реального датасета взять в обучение.",
        "synth_size_label": "Доля синтетики",
        "synth_size_help": "Только для режима single: какую долю синтетического датасета добавить.",
        "advanced_label": "Дополнительные параметры",
        "threshold_label": "Порог бинаризации маски",
        "threshold_help": "Порог вероятности (0...1), выше которого пиксель считается дефектом.",
        "lr_label": "Скорость обучения (learning rate)",
        "lr_help": "Шаг оптимизатора Adam. Меньше - стабильнее, но медленнее.",
        "seed_label": "Seed (зерно случайности)",
        "seed_help": "Фиксирует разбиение и перемешивание для воспроизводимости.",
        "preview_btn": "Предпросмотр датасета",
        "preview_counts": "Реальных изображений: {real}; синтетических: {synth}",
        "preview_samples": "Примеры пар 'изображение / маска'",
        "run_btn": "Запустить обучение",
        "running": "Идёт обучение... это может занять время на CPU.",
        "progress_fraction": "Доля синтетики {idx}/{total}: real={real}, synth={synth}",
        "done_in": "Готово за {sec} с",
        "results_table": "Метрики по долям",
        "chart_title": "IoU и Loss в зависимости от доли синтетики",
        "chart_synth": "Доля синтетики",
        "chart_iou": "IoU",
        "chart_loss": "Loss",
        "samples_title": "Примеры предсказаний (зелёный - истина, красный - предсказание)",
        "err_path_missing": "Путь не найден или в нём нет подпапки images/: {path}",
        "err_empty": "В датасете не найдено изображений: {path}",
        # - контуры -
        "contours_header": "Извлечение контуров дефекта",
        "upload_label": "Загрузите bitmap дефекта",
        "upload_help": "Чёрно-белая маска (bitmap) дефекта; контуры ищутся по светлым областям.",
        "extract_label": "Что извлечь",
        "extract_help": "Контуры (полигоны), ограничивающие прямоугольники или и то и другое.",
        "extract_contours": "Контуры",
        "extract_bbox": "Ограничивающие прямоугольники",
        "extract_both": "Контуры и прямоугольники",
        "extract_btn": "Извлечь",
        "contours_found": "Найдено контуров: {n}",
        "bbox_found": "Найдено прямоугольников: {n}",
        "overlay_caption": "Контуры (зелёные) / прямоугольники (синие) поверх маски",
        "download_csv": "Скачать CSV",
        # - результаты -
        "results_header": "Просмотр сохранённых результатов",
        "results_source_label": "Файл результатов",
        "results_help": "JSON-файлы из папки results/, которые пишет обучение.",
        "results_none": "В папке results/ нет JSON-файлов. Сначала запустите обучение.",
        "results_summary": "Прогон: {n} точек, суммарное время {sec} с",
        # - справка -
        "help_header": "Как пользоваться программой",
    },
    "en": {
        "app_title": "Synthetic Data Verifier - defect segmentation",
        "app_caption": "Train a U-Net on a mix of real and synthetic data",
        "language_label": "Interface language",
        "nav_help": "Help",
        "nav_train": "Training",
        "nav_contours": "Contours",
        "nav_results": "Results",
        "train_header": "Training and verification",
        "real_path_label": "Real data folder",
        "real_path_help": "Path to a dataset with images/ and bitmaps/ subfolders (real photos).",
        "synth_path_label": "Synthetic data folder",
        "synth_path_help": "Path to the synthetic dataset (same images/ + bitmaps/ structure).",
        "mode_label": "Mode",
        "mode_help": "sweep - vary synthetic fraction 0...100%; single - one run with fixed fractions.",
        "mode_sweep": "Sweep synthetic fraction (sweep)",
        "mode_single": "Single run (single)",
        "epochs_label": "Number of epochs",
        "epochs_help": "How many times the model passes over the training set per run.",
        "batch_label": "Batch size",
        "batch_help": "How many images are processed per training step.",
        "test_size_label": "Test set size",
        "test_size_help": "How many real images to hold out for testing (disjoint from training).",
        "max_images_label": "Image limit (0 = no limit)",
        "max_images_help": "Truncate datasets to N images - for fast CPU demo runs.",
        "real_size_label": "Real data fraction",
        "real_size_help": "single mode only: fraction of the real dataset used for training.",
        "synth_size_label": "Synthetic fraction",
        "synth_size_help": "single mode only: fraction of the synthetic dataset to add.",
        "advanced_label": "Advanced parameters",
        "threshold_label": "Mask binarization threshold",
        "threshold_help": "Probability threshold (0...1) above which a pixel counts as a defect.",
        "lr_label": "Learning rate",
        "lr_help": "Adam optimizer step. Smaller is more stable but slower.",
        "seed_label": "Seed (random seed)",
        "seed_help": "Fixes the split and shuffling for reproducibility.",
        "preview_btn": "Preview dataset",
        "preview_counts": "Real images: {real}; synthetic: {synth}",
        "preview_samples": "Sample image / mask pairs",
        "run_btn": "Start training",
        "running": "Training... this may take a while on CPU.",
        "progress_fraction": "Synthetic fraction {idx}/{total}: real={real}, synth={synth}",
        "done_in": "Done in {sec} s",
        "results_table": "Metrics by fraction",
        "chart_title": "IoU and Loss vs synthetic fraction",
        "chart_synth": "Synthetic fraction",
        "chart_iou": "IoU",
        "chart_loss": "Loss",
        "samples_title": "Prediction samples (green - truth, red - prediction)",
        "err_path_missing": "Path not found or has no images/ subfolder: {path}",
        "err_empty": "No images found in dataset: {path}",
        "contours_header": "Defect contour extraction",
        "upload_label": "Upload a defect bitmap",
        "upload_help": "Black-and-white defect mask (bitmap); contours are found on bright areas.",
        "extract_label": "What to extract",
        "extract_help": "Contours (polygons), bounding boxes, or both.",
        "extract_contours": "Contours",
        "extract_bbox": "Bounding boxes",
        "extract_both": "Contours and boxes",
        "extract_btn": "Extract",
        "contours_found": "Contours found: {n}",
        "bbox_found": "Boxes found: {n}",
        "overlay_caption": "Contours (green) / boxes (blue) over the mask",
        "download_csv": "Download CSV",
        "results_header": "View saved results",
        "results_source_label": "Results file",
        "results_help": "JSON files from the results/ folder written by training.",
        "results_none": "No JSON files in results/. Run training first.",
        "results_summary": "Run: {n} points, total time {sec} s",
        "help_header": "How to use this program",
    },
}

# Большие справочные блоки храним отдельно (markdown).
HELP_MD = {
    "ru": """
### О программе
Инструмент проверяет, **помогает ли синтетика** обучению модели сегментации
дефектов на алюминиевых листах. Модель (U-Net) обучается на смеси реальных и
синтетических изображений, после чего измеряется качество (IoU) на отложенном
**реальном** тесте.

Датасет - это папка с двумя подпапками:
- `images/` - снимки (`image0.jpg`, `image1.jpg`, ...);
- `bitmaps/` - маски дефектов (`bitmap0.jpg`, ...), парные по номеру.

### Вкладка 'Обучение'
- **Папка реальных данных / Папка синтетических данных** - пути к датасетам
  (каждый со своими `images/` и `bitmaps/`).
- **Режим** - комбобокс:
  - *Перебор доли синтетики (sweep)* - обучает 11 моделей с долей синтетики
    0%, 10%, ..., 100% и строит график зависимости IoU/Loss от доли.
  - *Один прогон (single)* - одна модель с заданными вручную долями.
- **Количество эпох** - число проходов по обучающей выборке.
- **Размер батча** - сколько изображений за один шаг.
- **Размер тестовой выборки** - сколько реальных изображений отложить под тест
  (они не попадают в обучение - без утечки).
- **Лимит изображений** - обрезает датасеты до N штук для быстрых демо на CPU
  (0 - без ограничения).
- **Доля реальных / Доля синтетики** (только *single*) - ползунки 0...1.
- **Дополнительные параметры**: порог бинаризации маски, learning rate, seed.

Кнопка **Запустить обучение** показывает прогресс, таблицу метрик, график и
примеры предсказаний (зелёный - истинная маска, красный - предсказанная).

### Вкладка 'Контуры'
Загрузите bitmap дефекта и выберите в комбобоксе **Что извлечь**: контуры,
ограничивающие прямоугольники или и то и другое. Результат - наложение на маску
и CSV для скачивания.

### Вкладка 'Результаты'
Откройте сохранённый JSON из папки `results/` и посмотрите график IoU/Loss по
долям синтетики без повторного обучения.

### Язык
Селектор **Язык интерфейса** в сайдбаре мгновенно переводит весь интерфейс,
включая пункты комбобоксов и подсказки.
""",
    "en": """
### About
This tool checks **whether synthetic data helps** train a defect-segmentation
model for aluminium sheets. A U-Net is trained on a mix of real and synthetic
images, then quality (IoU) is measured on a held-out **real** test set.

A dataset is a folder with two subfolders:
- `images/` - photos (`image0.jpg`, `image1.jpg`, ...);
- `bitmaps/` - defect masks (`bitmap0.jpg`, ...), paired by index.

### "Training" tab
- **Real / Synthetic data folder** - paths to datasets (each with its own
  `images/` and `bitmaps/`).
- **Mode** - combobox:
  - *Sweep synthetic fraction (sweep)* - trains 11 models with synthetic share
    0%, 10%, ..., 100% and plots IoU/Loss vs fraction.
  - *Single run (single)* - one model with manually set fractions.
- **Number of epochs** - passes over the training set.
- **Batch size** - images per training step.
- **Test set size** - real images held out for testing (excluded from training,
  so there is no leakage).
- **Image limit** - truncate datasets to N images for fast CPU demos
  (0 - no limit).
- **Real / Synthetic fraction** (*single* only) - sliders 0...1.
- **Advanced parameters**: mask binarization threshold, learning rate, seed.

The **Start training** button shows progress, a metrics table, a chart and
prediction samples (green - ground-truth mask, red - predicted).

### "Contours" tab
Upload a defect bitmap and pick in the **What to extract** combobox: contours,
bounding boxes, or both. The result is an overlay on the mask plus a CSV download.

### "Results" tab
Open a saved JSON from the `results/` folder and view the IoU/Loss-vs-fraction
chart without retraining.

### Language
The **Interface language** selector in the sidebar instantly translates the whole
UI, including combobox options and tooltips.
""",
}


def current_lang() -> str:
    return st.session_state.get("lang", DEFAULT_LANG)


def t(key: str, **kwargs) -> str:
    lang = current_lang()
    text = TRANSLATIONS.get(lang, {}).get(key)
    if text is None:
        text = TRANSLATIONS[DEFAULT_LANG].get(key, key)
    return text.format(**kwargs) if kwargs else text


def help_md() -> str:
    return HELP_MD.get(current_lang(), HELP_MD[DEFAULT_LANG])


def language_selector():
    st.sidebar.selectbox(
        t("language_label"),
        options=list(LANGUAGES.keys()),
        format_func=lambda code: LANGUAGES[code],
        key="lang",
    )
