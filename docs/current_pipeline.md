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

    H["Извлечение исторических фактов<br/><b>gpt-5.6-sol</b><br/>Tooken Responses API<br/>до 120000 символов контекста"]
    H --> I["Строгая JSON Schema<br/>локальная валидация + correction retry<br/>проверка ссылок на Wikipedia"]
    I --> J["Структурированные факты<br/>analysis.json + constraints.json"]

    J --> K
    G --> K

    K["Создание промпта<br/><b>gpt-5.6-sol</b><br/>Tooken Responses API"]
    K --> L["Основной prompt на английском<br/>negative prompt<br/>prompt.json"]

    L --> M["Генерация панорамы<br/><b>gpt-image-2</b><br/>Tooken Images API<br/>явный equirectangular 360° prompt"]

    M --> N["Landscape 1536×1024<br/>центрирование в 2:1 → 2048×1024<br/>original.png + panorama.png"]

    N --> O["Техническая проверка<br/><b>Модель не используется</b><br/>Pillow"]
    O --> O1["Наличие, декодирование,<br/>размеры и 2:1"]

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
| Извлечение исторических фактов | `gpt-5.6-sol` | Tooken Club, Responses API |
| Создание промпта | `gpt-5.6-sol` | Tooken Club, Responses API |
| Генерация панорамы | `gpt-image-2` | Tooken Club, Images API |
| Анализ референсов | `Qwen/Qwen2.5-VL-3B-Instruct` | Kaggle, T4 |
| Визуальная и историческая проверка | `Qwen/Qwen2.5-VL-3B-Instruct` | Kaggle, T4 |

Число мультимодальных проверок задаётся `pipeline.visual_validation_runs`: `0` отключает
их, значения `1–5` запускают указанное количество независимых проверок каждой попытки.
Детерминированную техническую проверку можно отключить параметром
`pipeline.technical_validation_enabled` или флагом `--skip-technical-validation`.

## Этапы без моделей

- Поиск и парсинг Википедии — `WikipediaInformationProvider` и MediaWiki API.
- Проверка файлов референсов — Pillow.
- Проверка наличия, читаемости, размеров и 2:1 — Pillow.
- Создание perspective-кадров — локальная сферическая проекция.
- Решение о повторной генерации — `RetryController`, максимум три полные генерации.
- Сохранение и восстановление — JSON-артефакты, `state.json` и параметр `--resume`.

Kaggle-провайдеры `Qwen/Qwen2.5-7B-Instruct`, SDXL и Stable Diffusion 3.5 Medium
остаются зарегистрированными альтернативами и выбираются независимо в YAML или Admin UI.
