# Historical 360° Panorama Pipeline

Расширяемый конвейер прямой генерации исторической equirectangular-панорамы:

`Wikipedia → структурированные факты → prompt → SD 3.5 Medium → Qwen2.5-VL + технические проверки → panorama.png`

Текущая конфигурация использует только MediaWiki API как исторический источник. Две
текстовые стадии на `Qwen/Qwen2.5-7B-Instruct` в 4-bit NF4, мультимодальная проверка на
`Qwen/Qwen2.5-VL-3B-Instruct` и генерация `Stable Diffusion 3.5 Medium` выполняются в отдельных Kaggle kernels;
модели локально не устанавливаются. Реализации выбираются через фабричные реестры в
`historical_panorama.factories`.

## Этапы

1. Wikipedia-провайдер разрешает redirects и disambiguation, очищает статьи с сохранением
   разделов, извлекает infobox и выбирает не более `max_related_articles` связанных статей.
2. LLM извлекает факты с уровнем `supported`, `inferred` или `unknown`, ссылкой на статью и
   раздел. `lm-format-enforcer` ограничивает допустимые токены JSON Schema непосредственно
   во время генерации, после чего ответ повторно проверяется; kernel делает одну попытку
   исправления семантических нарушений. Модель выбирает `source_id`, а статья и раздел
   подставляются детерминированно, поэтому она не может выдумать название источника.
3. Необязательные референсы проверяются локально, а Qwen2.5-VL описывает только категории
   из `use_for`, исключая элементы из `do_not_copy`. Повреждённые референсы пропускаются.
4. Вторая LLM получает только структурированный анализ, ограничения и сведения о
   референсах — необработанный текст Wikipedia в генератор prompt не передаётся.
5. Stable Diffusion 3.5 Medium непосредственно создаёт PNG 2:1. Blender, cubemap и
   3D-сцена не используются. Для T4 включены FP16, model CPU offload и VAE tiling.
6. Детерминированный валидатор проверяет файл, размеры, 2:1, пустые/чёрные/повреждённые
   области, резкость и левый/правый шов.
7. Панорама преобразуется в восемь горизонтальных perspective-кадров и два кадра с pitch
   `+60°/-60°`.
8. Qwen2.5-VL в Kaggle получает perspective-кадры, структурированные факты и ограничения и
   возвращает проблемы по строгой JSON Schema. При недоступности kernel сохраняется статус
   `visual_validation_unavailable`, а технические проверки продолжаются.
9. При технической ошибке controller добавляет конкретную коррекцию в prompt и повторяет
   полную генерацию, но не более трёх раз. После исчерпания попыток статус —
   `manual_review_required`.

## Установка и настройка

Нужны Python 3.11+, аккаунт Kaggle и GPU-квота Kaggle Notebooks.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp config.example.yaml config.yaml
cp .env.example .env
```

В `.env` укажите `KAGGLE_USERNAME` и `KAGGLE_API_TOKEN` (также поддерживается legacy
`KAGGLE_KEY`). `.env` исключён из Git.

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

Откройте `http://127.0.0.1:8080`. По умолчанию сервер доступен только локально. Kaggle и
Hugging Face токены передаются в окружение отдельного процесса запуска и не записываются в
YAML, `state.json` или браузерное хранилище. Служебные данные запусков находятся в
`.web-runs/`.

Фотографии можно прикрепить только тогда, когда выбранный генератор объявляет
`supports_image_conditioning = True`. Интерфейс блокирует загрузку для text-only моделей,
а worker повторно проверяет capability перед внешними запросами. До четырёх JPEG, PNG или
WebP по 8 МБ сохраняются внутри каталога запуска и передаются генератору через
`PromptResult.reference_paths`. Встроенные SD 3.5 и SDXL kernels пока являются text-only.

Если порт `8080` уже занят, укажите другой:

```bash
historical-panorama-web --config config.yaml --port 8081
```

Список вариантов централизован в `historical_panorama.web`, а сами реализации по-прежнему
создаются фабриками. Поэтому будущий API-провайдер (например, GPT) добавляется как новая
фабрика и описание варианта, не меняя страницу мониторинга и pipeline.

VPN влияет только на реальные запросы Wikipedia/Kaggle. Unit-тесты используют моки и не
требуют сети или GPU.

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

Алгоритмы детерминированно проверяют входные файлы, JSON Schema, размеры/2:1, области
изображения, резкость, характеристики шва, предел попыток и создают perspective-кадры.
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
