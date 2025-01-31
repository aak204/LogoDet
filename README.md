# Распознавание логотипов одежды

## Описание проекта

Данный проект реализует систему распознавания логотипов брендов одежды с использованием методов машинного обучения и глубокого обучения. В качестве демонстрационного прототипа использовались только бренды одежды из датасета [LogoDet-3K]([https://github.com/xyz/LogoDet-3K](https://paperswithcode.com/dataset/logodet-3k)). Система включает в себя:

- Загрузку и обработку данных.
- Обучение модели на основе сверточной нейронной сети.
- Валидацию и оценку модели.
- Верификацию логотипов на новых изображениях.

Для достижения баланса между скоростью работы и качеством результатов была выбрана нейросеть LCNet-050. В дальнейшем планируется использование более сложных архитектур для повышения точности распознавания.

## Структура проекта

```
logo_recognition/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   ├── validate.py
│   ├── utils.py
│   └── main.py
```

## Установка

1. **Клонируйте репозиторий:**

    ```bash
    git clone https://github.com/ваш-ник/logo_recognition.git
    cd logo_recognition
    ```

2. **Создайте и активируйте виртуальное окружение (рекомендуется):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Для Unix
    venv\Scripts\activate     # Для Windows
    ```

3. **Установите необходимые зависимости:**

    ```bash
    pip install -r requirements.txt
    ```

## Подготовка данных

Скачайте датасет [LogoDet-3K](https://paperswithcode.com/dataset/logodet-3k) и разместите его в папке `data/LogoDet-3K/Clothes/`. Структура должна соответствовать следующему примеру:

```
data/
└── LogoDet-3K/
    └── Clothes/
        ├── BrandA/
        │   ├── image1.jpg
        │   ├── image1.xml
        │   └── ...
        ├── BrandB/
        │   ├── image2.jpg
        │   ├── image2.xml
        │   └── ...
        └── ...
```

## Использование

### Обучение модели

Для начала обучения модели выполните скрипт `main.py` с параметром `train`:

```bash
python src/main.py train
```

### Валидация модели

После завершения обучения модель сохраняется в файл `lcnet_logo_encoder.pth`. Для валидации модели выполните:

```bash
python src/main.py validate
```

### Верификация логотипов

Для проверки логотипа на изображении используйте функцию `verify_logo` в `main.py` или создайте отдельный скрипт, который будет загружать модель и выполнять верификацию.

### Пример использования

В `main.py` предоставлены примеры загрузки модели и верификации логотипов:

```python
# Загрузка модели
model = LogoEncoder(emb_dim=EMB_DIM)
ckpt = torch.load("lcnet_logo_encoder.pth", map_location='cuda')
model.load_state_dict(ckpt)
model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
model = model.cuda().eval()

# Сборка эталонных эмбеддингов
brandX_imgs = [
    "data/LogoDet-3K/Clothes/Berghaus/0.jpg",
    "data/LogoDet-3K/Clothes/Berghaus/1.jpg"
]
reference_embs = collect_reference_embeddings(brandX_imgs, model)

# Верификация логотипа
query_img = Image.open("data/LogoDet-3K/Clothes/Berghaus/10.jpg").convert("RGB")
is_brandX, sim_val = verify_logo(query_img, reference_embs, model, threshold=0.65)
print(f"Is brand Berghaus? {is_brandX}. Mean similarity = {sim_val:.2f}")
```

## Файлы и папки

- **src/dataset.py**: реализация класса `LogoDataset` для загрузки и обработки данных.
- **src/model.py**: определение архитектуры модели `LogoEncoder`.
- **src/loss.py**: реализация функции потерь `SupConLoss`.
- **src/train.py**: функции для обучения и валидации модели.
- **src/validate.py**: функции для оценки модели на валидационном наборе.
- **src/utils.py**: вспомогательные функции, такие как `verify_logo`, `invert_dict`, и `show_dataset_samples_with_bbox`.
- **src/main.py**: основной скрипт для запуска обучения, валидации и верификации.
- **requirements.txt**: список зависимостей проекта.

## Зависимости

Основные зависимости проекта перечислены в `requirements.txt`:

```
torch
torchvision
timm
numpy
Pillow
matplotlib
```

Установите их с помощью команды:

```bash
pip install -r requirements.txt
```

## Лицензия

Этот проект распространяется под лицензией MIT. Подробности см. в файле [LICENSE](LICENSE).
