# Current Historical Panorama Pipeline

```mermaid
flowchart TD
    A["Вход<br/>Название исторического события"] --> B

    B["Поиск исторической информации<br/><b>Модель не используется</b><br/>WikipediaInformationProvider<br/>MediaWiki API русской Википедии"]
    B --> C["Очищенные статьи и разделы<br/>research.json"]

    R["0–4 референсных изображения"] --> D
    C --> D

    D["Проверка референсов<br/><b>Модель не используется</b><br/>Pillow + локальные алгоритмы<br/>формат, существование, читаемость"]
    D --> E{"Нужно описать<br/>референсы?"}

    E -->|Да| F["Анализ референсов<br/><b>Qwen/Qwen2.5-VL-3B-Instruct</b><br/>Kaggle GPU T4"]
    E -->|Нет| G
    F --> G["references.json"]

    C --> H
    G --> H

    H["Извлечение исторических фактов<br/><b>Qwen/Qwen2.5-7B-Instruct</b><br/>4-bit NF4, Kaggle GPU T4"]
    H --> I["Строгая JSON Schema<br/>lm-format-enforcer"]
    I --> J["Структурированные факты<br/>analysis.json + constraints.json"]

    J --> K
    G --> K

    K["Создание промпта<br/><b>Qwen/Qwen2.5-7B-Instruct</b><br/>4-bit NF4, Kaggle GPU T4"]
    K --> L["Основной prompt на английском<br/>negative prompt<br/>prompt.json"]

    L --> M["Генерация панорамы<br/><b>stabilityai/stable-diffusion-3.5-medium</b><br/>StableDiffusion3Pipeline<br/>FP16 + CPU offload<br/>Kaggle GPU T4"]
    HF["Hugging Face gated access<br/>Kaggle Secret: HF_TOKEN"] --> M

    M --> N["Прямая equirectangular-генерация<br/>1536×768 → 2048×1024<br/>panorama.png"]

    N --> O["Техническая проверка<br/><b>Модель не используется</b><br/>Pillow + NumPy + OpenCV"]
    O --> O1["2:1 и размеры"]
    O --> O2["Чёрные/пустые области"]
    O --> O3["Размытие"]
    O --> O4["Цвет и яркость шва"]
    O --> O5["Непрерывность горизонта"]

    N --> P["Преобразование в perspective-кадры<br/><b>Модель не используется</b><br/>Локальная сферическая проекция"]
    P --> P1["Yaw: 0°, 45° ... 315°<br/>Pitch: 0°, +60°, −60°"]

    P1 --> Q
    J --> Q
    G --> Q

    Q["Визуальная и историческая проверка<br/><b>Qwen/Qwen2.5-VL-3B-Instruct</b><br/>Kaggle GPU T4"]
    Q --> Q1["Дефекты людей и архитектуры"]
    Q --> Q2["Современные объекты и анахронизмы"]
    Q --> Q3["Проверка must_include"]
    Q --> Q4["Проверка must_not_include"]
    Q --> Q5["Согласованность соседних кадров"]

    O1 --> S["RetryController<br/><b>Модель не используется</b>"]
    O2 --> S
    O3 --> S
    O4 --> S
    O5 --> S
    Q1 --> S
    Q2 --> S
    Q3 --> S
    Q4 --> S
    Q5 --> S

    S --> T{"Результат допустим?"}
    T -->|Да| U["accepted<br/>panorama.png<br/>manifest.json<br/>final_report.json"]
    T -->|Нет, осталось менее 3 попыток| V["Добавление исправлений в prompt<br/>новый seed"]
    V --> M
    T -->|Нет, попытки исчерпаны| W["manual_review_required"]
```

## Используемые модели

| Назначение | Модель | Запуск |
|---|---|---|
| Извлечение исторических фактов | `Qwen/Qwen2.5-7B-Instruct` | Kaggle, T4, 4-bit NF4 |
| Создание промпта | `Qwen/Qwen2.5-7B-Instruct` | Kaggle, T4, 4-bit NF4 |
| Генерация панорамы | `stabilityai/stable-diffusion-3.5-medium` | Kaggle, T4, FP16 + CPU offload |
| Анализ референсов | `Qwen/Qwen2.5-VL-3B-Instruct` | Kaggle, T4 |
| Визуальная и историческая проверка | `Qwen/Qwen2.5-VL-3B-Instruct` | Kaggle, T4 |

## Этапы без моделей

- Поиск и парсинг Википедии — `WikipediaInformationProvider` и MediaWiki API.
- Проверка файлов референсов — Pillow.
- Проверка размеров, размытия и шва — Pillow, NumPy и OpenCV.
- Создание perspective-кадров — локальная сферическая проекция.
- Решение о повторной генерации — `RetryController`, максимум три полные генерации.
- Сохранение и восстановление — JSON-артефакты, `state.json` и параметр `--resume`.
