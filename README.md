# Historical 360° Panorama Pipeline

Расширяемый конвейер прямой генерации исторической equirectangular-панорамы:

`Wikipedia → структурированные факты → prompt → генератор панорамы → визуальные и технические проверки → panorama.png`

Текущая конфигурация использует MediaWiki API как исторический источник, `gpt-5.6-sol`
через OpenAI-совместимый Tooken Club для двух текстовых стадий и `gpt-image-2` для
изображения. Мультимодальная проверка остаётся на `Qwen/Qwen2.5-VL-3B-Instruct` в Kaggle.
Kaggle-реализации Qwen, SDXL и Stable Diffusion 3.5 сохранены как альтернативы. Все
реализации выбираются через фабричные реестры в `historical_panorama.factories`.

## Этапы

1. Wikipedia-провайдер разрешает redirects и disambiguation, очищает статьи с сохранением
   разделов, извлекает infobox и выбирает не более `max_related_articles` связанных статей.
2. LLM извлекает факты с уровнем `supported`, `inferred` или `unknown`, ссылкой на статью и
   раздел. Для Tooken ответ проверяется локально по строгой JSON Schema и при ошибке один
   раз запрашивается исправленный JSON; ссылки дополнительно сверяются с реально переданными
   разделами Wikipedia. Kaggle-вариант использует `lm-format-enforcer` и детерминированную
   подстановку источника по `source_id`.
3. Необязательные референсы проверяются локально, а Qwen2.5-VL описывает только категории
   из `use_for`, исключая элементы из `do_not_copy`. Повреждённые референсы пропускаются.
4. Вторая LLM получает только структурированный анализ, ограничения и сведения о
   референсах — необработанный текст Wikipedia в генератор prompt не передаётся.
5. `gpt-image-2` получает явные требования seamless equirectangular 360°, `360° × 180°`
   и 2:1. Поддерживаемый API landscape-кадр центрируется и приводится к точному
   `2048×1024`; исходный ответ сохраняется как `original.png`. При выборе Kaggle вместо
   этого используется прямая SD 3.5/SDXL генерация. Blender, cubemap и 3D не используются.
6. Детерминированный валидатор проверяет наличие и читаемость файла, размеры и 2:1.
7. Панорама преобразуется в восемь горизонтальных perspective-кадров и два кадра с pitch
   `+60°/-60°`.
8. Qwen2.5-VL в Kaggle получает perspective-кадры, структурированные факты и ограничения и
   возвращает проблемы по строгой JSON Schema. При недоступности kernel сохраняется статус
   `visual_validation_unavailable`, а технические проверки продолжаются.
9. При технической ошибке controller добавляет конкретную коррекцию в prompt и повторяет
   полную генерацию, но не более трёх раз. После исчерпания попыток статус —
   `manual_review_required`.

Число мультимодальных проверок одной попытки задаётся в YAML:

```yaml
pipeline:
  technical_validation_enabled: true  # false = отключить файл/размеры/2:1
  visual_validation_runs: 1  # 0 = выключить; допустимый диапазон при включении: 1-5
```

Каждый запуск получает отдельный request ID и сохраняется как
`visual_validation_01.json`, `visual_validation_02.json` и т. д. Их агрегированный результат
остаётся в `visual_validation.json`. Детерминированная техническая проверка управляется
отдельным параметром `technical_validation_enabled`.

Для разового запуска значение можно переопределить без изменения YAML:

```bash
historical-panorama --config config.yaml --visual-validation-runs 0 "Название события"
historical-panorama --config config.yaml --visual-validation-runs 3 "Название события"
historical-panorama --config config.yaml --skip-technical-validation "Название события"
```

## Установка и настройка

Нужны Python 3.11+, аккаунт Kaggle и GPU-квота Kaggle Notebooks.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp config.example.yaml config.yaml
cp .env.example .env
```

В `.env` укажите `GPT_TOKEN` для Tooken Club. Для Kaggle-визуальной проверки также нужны
`KAGGLE_USERNAME` и `KAGGLE_API_TOKEN` (поддерживается legacy `KAGGLE_KEY`). `.env`
исключён из Git.

`stabilityai/stable-diffusion-3.5-medium` — gated-модель. Перед первым запуском:

1. войдите в Hugging Face и примите условия на странице модели;
2. создайте read-token Hugging Face;
3. добавьте `HF_TOKEN=hf_...` в локальный `.env`.

Kaggle API не переносит разрешения Kaggle Secrets в загружаемые script kernels стабильно.
Поэтому pipeline передаёт `HF_TOKEN` внутрь приватного kernel-запроса. Значение не попадает
в `config.yaml`, отчёты или логи, но присутствует в исходнике приватной версии Kaggle.
Используйте fine-grained read-token, не делайте notebook публичным и отзовите токен при
подозрении на утечку.

Проверка подключения:

```bash
historical-panorama --config config.yaml --check-connections
```

При нестабильном IPv6 можно включить `kaggle.force_ipv4: true`. Если зависает только
необязательный preflight, `--skip-connection-check` пропускает его; авторизация всё равно
проверяется первой фактической командой Kaggle. Одиночные ошибки polling статуса повторяются
до пяти раз и не считаются немедленной ошибкой удалённого kernel.

Полный запуск:

```bash
historical-panorama --config config.yaml "Строительство крепости Свияжск в 1551 году"
```

### Веб-интерфейс администратора

Администратор может независимо выбрать провайдера для поиска, извлечения фактов,
построения промпта, генерации и визуальной проверки, а затем наблюдать прогресс, попытки,
журнал ошибок и итоговую панораму:

```bash
historical-panorama-web --config config.yaml
```

Откройте `http://127.0.0.1:8080`. По умолчанию сервер доступен только локально. Kaggle,
Hugging Face и Tooken токены передаются в окружение отдельного процесса запуска и не
записываются в YAML, `state.json` или браузерное хранилище. Служебные данные находятся в
`.web-runs/`.

Фотографии можно прикрепить только тогда, когда выбранный генератор объявляет
`supports_image_conditioning = True`. Интерфейс блокирует загрузку для text-only моделей,
а worker повторно проверяет capability перед внешними запросами. До четырёх JPEG, PNG или
WebP по 8 МБ сохраняются внутри каталога запуска и передаются генератору через
`PromptResult.reference_paths`. Встроенные SD 3.5, SDXL и `gpt-image-2` провайдеры пока
являются text-only.

Если порт `8080` уже занят, укажите другой:

```bash
historical-panorama-web --config config.yaml --port 8081
```

Список вариантов централизован в `historical_panorama.web`, а реализации создаются
фабриками, поэтому Tooken и Kaggle можно независимо выбирать для каждой стадии.

VPN влияет только на реальные запросы Wikipedia, Tooken и Kaggle. Unit-тесты используют
моки и не требуют сети, баланса или GPU.

## Выбор Tooken или Kaggle

Текущие фабрики `tooken` используют `https://tooken.club/v1`. Для извлечения фактов
доступен увеличенный лимит `max_context_characters: 120000`; это не жёсткий model context
window, а безопасный предел объёма очищенных материалов Wikipedia в одном запросе.

Чтобы вернуть Kaggle-текстовые модели, замените соответствующие секции на:

```yaml
fact_extractor:
  provider: kaggle
  kernel_slug: historical-panorama-fact-extractor
  model_id: Qwen/Qwen2.5-7B-Instruct
  accelerator: NvidiaTeslaT4
  load_in_4bit: true
  max_new_tokens: 1400

prompt_builder:
  provider: kaggle
  kernel_slug: historical-panorama-prompt-builder
  model_id: Qwen/Qwen2.5-7B-Instruct
  accelerator: NvidiaTeslaT4
  load_in_4bit: true
```

Для возврата генерации SD 3.5:

```yaml
image_generator:
  provider: kaggle_sd35
  kernel_slug: historical-panorama-sd35-generator
  model_id: stabilityai/stable-diffusion-3.5-medium
  hf_token_env: HF_TOKEN
  accelerator: NvidiaTeslaT4
  width: 1536
  height: 768
  output_width: 2048
  output_height: 1024
  steps: 28
  guidance_scale: 4.5
  seed: 42
```

## Референсы

Можно передать JSON-файл с 0–4 элементами:

```json
[
  {
    "path": "reference.jpg",
    "use_for": ["clothing", "weapon_shape"],
    "do_not_copy": ["background", "composition"]
  }
]
```

```bash
historical-panorama --config config.yaml --references references.json "Название события"
```

Или повторить `--reference` с inline JSON. Повреждённые, отсутствующие и неподдерживаемые
файлы пропускаются с записью причины в отчёт. При наличии `use_for` валидный референс
анализируется отдельным Qwen2.5-VL kernel; исходное разрешение перед отправкой уменьшается.

## Сохранение и восстановление

Каждый этап атомарно обновляет `state.json`. В каталоге запуска сохраняются:

- `research.json`, `analysis.json`, `constraints.json`, `references.json`;
- `prompt.json`, `structured_description.json`, `used_facts.json`, `sources.json`;
- `attempts/attempt_NN/` с prompt, panorama, параметрами генерации, perspective-кадрами и
  техническим/визуальным отчётами;
- итоговые `panorama.png`, `manifest.json` и `final_report.json`.

Продолжение прерванного запуска:

```bash
historical-panorama --config config.yaml --resume runs/<run-directory>
```

Успешно сохранённые Wikipedia-материалы, анализ, prompt и готовые попытки повторно не
запрашиваются. CLI возвращает код `2`, если итог требует ручной проверки.

## Расширение

Основные протоколы находятся в `historical_panorama.interfaces`: `InformationProvider`,
`HistoricalFactExtractor`, `PromptBuilder`, `ImageGenerator`, `VisualValidator` и
`ReferenceAnalyzer`. Встроенные
реестры:

- `information_provider_factories`;
- `fact_extractor_factories`;
- `prompt_builder_factories`;
- `image_generator_factories`;
- `visual_validator_factories`.

Внешняя реализация регистрируется в модуле и подключается через `factory_modules` в YAML:

```python
from historical_panorama.factories import information_provider_factories

@information_provider_factories.register("my_wikipedia_cache")
def build_provider(config, context):
    return MyWikipediaCache(config["path"])
```

На текущем этапе исторический provider должен оставаться Wikipedia-only. Общий
`KaggleKernelRunner` создаётся лениво через `FactoryContext`, а проверка соединения
выполняется один раз.

## Что проверяет код, а что — модели

Алгоритмы детерминированно проверяют входные файлы, JSON Schema, размеры/2:1, предел
попыток и создают perspective-кадры.
Wikipedia очищается и ранжируется парсером.

Текстовая Qwen выделяет исторические факты и формирует финальное визуальное описание. SD 3.5 Medium
генерирует изображение. Qwen2.5-VL анализирует разрешённые свойства референсов и проверяет
perspective-кадры на визуальные дефекты и соответствие переданным ограничениям. Она не
используется как самостоятельный исторический источник.

## Ограничения

- SD 3.5 Medium не гарантирует идеальную сферическую геометрию и бесшовность; validator
  обнаруживает заметные дефекты, но не исправляет пиксели сам.
- Технические пороги являются эвристиками и могут потребовать настройки под другую модель.
- SD 3.5 Medium пока не поддерживает image conditioning и inpainting в текущей реализации;
  референсы влияют на prompt через текстовое описание Qwen2.5-VL.
- Мультимодальная оценка вероятностная и может пропустить дефект; строгая JSON Schema
  гарантирует структуру ответа, но не истинность суждения модели.
- Метаданные Wikipedia могут быть неполными; это записывается в `limitations`, после чего
  pipeline продолжает работу с доступным материалом.

## Тесты

```bash
pytest -q
```

Тесты не запускают Kaggle kernels и не расходуют GPU-квоту.
