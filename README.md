# Historical 360° Panorama Pipeline

Рабочий расширяемый конвейер:

`название события → веб-исследование → построение промпта → SDXL → PNG 2:1`

Текущая тестовая конфигурация использует MediaWiki API для поиска, мультиязычную
`Qwen/Qwen2.5-3B-Instruct` в Kaggle для построения промпта и SDXL в отдельном Kaggle GPU
kernel для генерации. Каждый блок зависит только от интерфейса в
`src/historical_panorama/interfaces.py`, поэтому его можно заменить на RAG, OpenAI API,
локальную модель или другой сервис без изменения оркестратора.

## Что происходит при запуске

1. Wikipedia-провайдер находит несколько статей и сохраняет факты и URL в `research.json`.
2. Первый приватный Kaggle kernel преобразует факты в англоязычный промпт с требованиями
   эквиректангуларной панорамы.
3. Второй приватный Kaggle GPU kernel запускает SDXL и возвращает `panorama.png`.
4. Локальный код проверяет статус каждого kernel, наличие ответа, совпадение `run_id` и
   соотношение сторон 2:1. Итог и источники сохраняются в `runs/<run-id>/`.

В логах явно видны: проверка аутентификации, принятие job сервисом Kaggle, переходы
статуса, скачивание и проверка ответа. Ошибка Kaggle CLI, авторизации, kernel, таймаут,
старый output или отсутствующий файл завершают запуск с понятным сообщением.

## Установка

Нужны Python 3.11+, аккаунт Kaggle и доступная GPU-квота Kaggle Notebooks.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pip install kaggle
cp config.example.yaml config.yaml
cp .env.example .env
```

В репозитории уже есть рабочий `config.yaml`; копировать example нужно только если вы
хотите вернуть настройки к исходным значениям.

В `.env` укажите `KAGGLE_USERNAME` и токен. Токен создаётся в настройках Kaggle API.
Предпочтителен текущий `KAGGLE_API_TOKEN`; поддержан и legacy `KAGGLE_KEY`. Файл `.env`
исключён из Git. Никогда не присылайте и не коммитьте токен в исходный код.

Сначала проверьте только подключение:

```bash
historical-panorama --config config.yaml --check-connections
```

Успех выглядит так:

```text
Kaggle: checking authentication for user ...
Kaggle: connection and authentication succeeded.
```

Запуск полного конвейера:

```bash
historical-panorama --config config.yaml "Строительство крепости Свияжск в 1551 году"
```

Первый запуск может быть долгим: два Kaggle job скачивают модели. По умолчанию общий
таймаут каждого job — 30 минут. Kernel имеют постоянные имена и при следующих запусках
создают новые версии, а не бесконечно новые проекты.

## Замена блоков

Новый поиск реализует `InformationProvider.research`, построитель — `PromptBuilder.build`,
генератор — `ImageGenerator.generate`. Затем фабрика регистрируется в одном из реестров из
`historical_panorama.factories`. Kaggle не является частью центрального pipeline — это лишь
две текущие реализации провайдеров.

Например, внешний модуль `my_project/providers.py` может зарегистрировать RAG-поиск:

```python
from historical_panorama.factories import information_provider_factories

@information_provider_factories.register("my_rag")
def build_my_rag(config, context):
    return MyRagProvider(index_url=config["index_url"])
```

Подключение и выбор выполняются только через YAML:

```yaml
factory_modules:
  - my_project.providers

search:
  provider: my_rag
  index_url: https://example.test/index
```

Аналогично используются `prompt_builder_factories` и `image_generator_factories`. Менять
`config.py` и `HistoricalPanoramaPipeline` при добавлении провайдера не нужно. Общие внешние
клиенты можно лениво получать через `FactoryContext`; встроенные Kaggle-фабрики используют
`context.kaggle_runner()`. Проверка подключения выполняется один раз для каждого созданного
сервиса.

Настройки моделей, размеров, seed, числа шагов и таймаутов вынесены в `config.yaml`.
Для prompt builder и SDXL явно запрашивается `NvidiaTeslaT4`; это исключает ошибки
несовместимости CUDA на старых GPU Kaggle.
Перед запуском SDXL промпт проходит quality gate: проверяются минимальная содержательность,
английский язык и наличие требований эквиректангуларной геометрии.
Генерация выполняется в 1536×768, затем результат увеличивается до 2048×1024. Это экономит
GPU-память, но апскейл не добавляет деталей; при достаточной Kaggle GPU можно увеличить
исходные размеры, сохраняя строгое отношение 2:1.

## Ограничение качества 360°

Формат результата совместим с просмотрщиками эквиректангуларных панорам: PNG и 2:1.
Однако базовый SDXL не даёт математической гарантии бесшовного стыка или корректной
проекции у полюсов. Промпт существенно помогает, но для production-качества стоит заменить
генератор на panorama-specific checkpoint/LoRA либо добавить отдельный seam/pole quality
gate. Архитектура позволяет сделать это без изменения поиска и prompt builder.

## Тесты

```bash
pytest -q
```

Unit-тесты не расходуют Kaggle GPU-квоту. Полный end-to-end запуск начинается только
явной CLI-командой после настройки токена.
