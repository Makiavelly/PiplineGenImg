const $ = (selector) => document.querySelector(selector);
let options = null;
let pollTimer = null;
let imageFiles = [];
let stagedMode = false;
let stagedModeAvailable = false;
let artifactRevision = '0';
let artifactJobId = null;

const stageHints = {
  search: 'Сбор материалов',
  fact_extractor: 'Структурирование',
  prompt_builder: 'Описание сцены',
  image_generator: 'Синтез изображения',
  visual_validator: 'Контроль качества',
};

async function api(path, init) {
  const response = await fetch(path, init);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || 'Ошибка запроса');
  return body;
}

function renderStages(data) {
  options = data;
  $('#stageList').innerHTML = data.stages.map((stage, index) => `
    <div class="stage-row">
      <span class="stage-index">${String(index + 1).padStart(2, '0')}</span>
      <div class="stage-name"><strong>${stage.label}</strong><small>${stageHints[stage.id]}</small></div>
      <div class="provider-select">
        <select data-stage="${stage.id}" aria-label="Провайдер: ${stage.label}">
          ${stage.providers.map(provider => `<option value="${provider.id}" ${provider.id === stage.selected ? 'selected' : ''}>${provider.label}</option>`).join('')}
        </select>
        <p data-description="${stage.id}"></p>
      </div>
    </div>
  `).join('');
  $('#visualValidationEnabled').checked = Boolean(data.visual_validation?.enabled ?? true);
  $('#visualValidationRuns').value = data.visual_validation?.runs || 1;
  $('#technicalValidationEnabled').checked = Boolean(data.technical_validation?.enabled ?? true);
  stagedModeAvailable = Boolean(data.demo?.available);
  stagedMode = false;
  document.querySelectorAll('[data-stage]').forEach(select => {
    select.addEventListener('change', () => {
      if (select.dataset.stage === 'visual_validator') {
        $('#visualValidationEnabled').checked = select.value !== 'unavailable';
      }
      updateDescription(select);
      updateFormDependencies();
    });
    updateDescription(select);
  });
  updateFormDependencies();
}

function updateDescription(select) {
  const stage = options.stages.find(item => item.id === select.dataset.stage);
  const provider = stage.providers.find(item => item.id === select.value);
  document.querySelector(`[data-description="${stage.id}"]`).textContent = provider.description;
}

function selections() {
  return Object.fromEntries([...document.querySelectorAll('[data-stage]')].map(select => [select.dataset.stage, select.value]));
}

function selectedImageProvider() {
  if (!options) return null;
  const stage = options.stages.find(item => item.id === 'image_generator');
  const providerId = selections().image_generator;
  return stage.providers.find(item => item.id === providerId) || null;
}

function canAttachImages() {
  if (stagedMode || Boolean(selectedImageProvider()?.supports_images)) return true;
  return $('#visualValidationEnabled').checked && selections().visual_validator === 'kaggle';
}

function updateFormDependencies() {
  const values = Object.values(selections());
  const usesKaggle = values.some(value => value.startsWith('kaggle'));
  const usesTooken = values.some(value => ['tooken', 'openai_compatible'].includes(value));
  $('#credentials').classList.toggle('hidden', !usesKaggle && !usesTooken);
  $('#hfToken').closest('label').classList.toggle('hidden', selections().image_generator !== 'kaggle_sd35');
  const providerSupportsImages = Boolean(selectedImageProvider()?.supports_images);
  const supportsImages = canAttachImages();
  $('#imageInput').disabled = !supportsImages;
  $('#dropzone').disabled = !supportsImages;
  $('#uploadCapability').textContent = stagedMode
    ? 'Для этой модели: use_for и do_not_copy можно оставить пустыми'
    : supportsImages
    ? (providerSupportsImages
      ? 'Изображения будут переданы непосредственно генератору'
      : 'Qwen2.5-VL извлечёт только детали use_for и передаст генератору их текстовое описание')
    : 'Ни генератор, ни выбранный визуальный анализатор не принимают изображения';
  if (!supportsImages && imageFiles.length) clearImages();
  const visualProvider = selections().visual_validator;
  const visualEnabled = $('#visualValidationEnabled').checked && visualProvider !== 'unavailable';
  $('#visualValidationRuns').disabled = !visualEnabled;
}

function resetForm() {
  if (!options) return;
  for (const stage of options.stages) {
    const select = document.querySelector(`[data-stage="${stage.id}"]`);
    select.value = stage.selected;
    updateDescription(select);
  }
  $('#visualValidationEnabled').checked = Boolean(options.visual_validation?.enabled ?? true);
  $('#visualValidationRuns').value = options.visual_validation?.runs || 1;
  $('#technicalValidationEnabled').checked = Boolean(options.technical_validation?.enabled ?? true);
  stagedMode = false;
  updateFormDependencies();
}

function clearImages() {
  imageFiles.forEach(item => URL.revokeObjectURL(item.url));
  imageFiles = [];
  $('#imageInput').value = '';
  renderImages();
}

function renderImages() {
  $('#uploadCount').textContent = `${imageFiles.length} / 4`;
  $('#uploadList').innerHTML = imageFiles.map((item, index) => `
    <div class="upload-item">
      <img src="${item.url}" alt="" data-upload-preview>
      <div class="upload-item-body">
        <strong>${escapeHtml(item.file.name)}</strong><small>${formatBytes(item.file.size)}</small>
        <label>Использовать для<input data-use-for="${index}" value="${escapeHtml(item.useFor)}" placeholder="clothing, weapon_shape"></label>
        <label>Не копировать<input data-do-not-copy="${index}" value="${escapeHtml(item.doNotCopy)}" placeholder="background, composition"></label>
      </div>
      <button type="button" data-remove-image="${index}" aria-label="Удалить фотографию">×</button>
    </div>
  `).join('');
  document.querySelectorAll('[data-use-for]').forEach(input => {
    input.addEventListener('input', () => { imageFiles[Number(input.dataset.useFor)].useFor = input.value; });
  });
  document.querySelectorAll('[data-do-not-copy]').forEach(input => {
    input.addEventListener('input', () => { imageFiles[Number(input.dataset.doNotCopy)].doNotCopy = input.value; });
  });
  document.querySelectorAll('[data-upload-preview]').forEach(preview => {
    preview.addEventListener('error', () => {
      preview.hidden = true;
      preview.closest('.upload-item').classList.add('preview-unavailable');
    }, { once: true });
  });
  document.querySelectorAll('[data-remove-image]').forEach(button => {
    button.addEventListener('click', () => {
      const index = Number(button.dataset.removeImage);
      URL.revokeObjectURL(imageFiles[index].url);
      imageFiles.splice(index, 1);
      renderImages();
    });
  });
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[character]));
}

function formatBytes(bytes) {
  return bytes < 1024 * 1024 ? `${Math.ceil(bytes / 1024)} КБ` : `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
}

function addImages(files) {
  $('#formError').textContent = '';
  if (!canAttachImages()) {
    $('#formError').textContent = 'Выбранная модель не поддерживает изображения в контексте.';
    return;
  }
  const accepted = ['image/jpeg', 'image/png', 'image/webp'];
  for (const file of files) {
    if (imageFiles.length >= 4) {
      $('#formError').textContent = 'Можно прикрепить не более четырёх фотографий.';
      break;
    }
    if (!accepted.includes(file.type)) {
      $('#formError').textContent = `Формат файла «${file.name}» не поддерживается.`;
      continue;
    }
    if (file.size > 8 * 1024 * 1024) {
      $('#formError').textContent = `Файл «${file.name}» превышает лимит 8 МБ.`;
      continue;
    }
    imageFiles.push({ file, url: URL.createObjectURL(file), useFor: '', doNotCopy: '' });
  }
  $('#imageInput').value = '';
  renderImages();
}

function fileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',', 2)[1]);
    reader.onerror = () => reject(new Error(`Не удалось прочитать файл «${file.name}»`));
    reader.readAsDataURL(file);
  });
}

const fieldLabels = {
  prompt: 'Основной промпт',
  negative_prompt: 'Negative prompt',
  identified_event: 'Определённое событие',
  date_or_period: 'Дата или период',
  place: 'Место',
  participants: 'Участники',
  event_type: 'Тип события',
  environment: 'Окружение',
  architecture: 'Архитектура',
  clothing: 'Одежда',
  weapons: 'Оружие',
  transport: 'Транспорт',
  everyday_objects: 'Предметы быта',
  natural_features: 'Природные особенности',
  visual_actions: 'Действия в кадре',
  unknown_or_disputed: 'Неизвестное или спорное',
  possible_anachronisms: 'Возможные анахронизмы',
  facts: 'Факты',
  constraints: 'Ограничения',
  must_include: 'Обязательно включить',
  may_include: 'Можно включить',
  must_not_include: 'Нельзя включать',
  do_not_over_specify: 'Не конкретизировать',
  metadata: 'Метаданные',
  sources: 'Источники',
  references: 'Изображения контекста',
  limitations: 'Ограничения данных',
  status: 'Статус',
  checks: 'Проверки',
  issues: 'Найденные проблемы',
  explanation: 'Пояснение',
  decision: 'Решение',
  generation: 'Генерация',
  provider: 'Провайдер',
  model_id: 'Модель',
  seed: 'Seed',
  width: 'Ширина',
  height: 'Высота',
  attempt: 'Попытка',
  attempts: 'Количество попыток',
  confidence: 'Достоверность',
  statement: 'Утверждение',
  category: 'Категория',
  query: 'Исходный запрос',
  primary_article: 'Основная статья',
  related_articles: 'Связанные статьи',
  title: 'Название',
  summary: 'Краткое описание',
  visual_reconstruction_notes: 'Указания для реконструкции',
  url: 'Ссылка',
  text: 'Текст источника',
  article: 'Статья',
  section: 'Раздел',
  relevance: 'Релевантность',
  is_primary: 'Основной источник',
  use_for: 'Использовать для',
  do_not_copy: 'Не копировать',
  valid: 'Файл проверен',
  format: 'Формат',
  description: 'Описание',
  description_status: 'Статус описания',
  reference_paths: 'Пути к референсам',
  model_output: 'Сформировано моделью',
  measured_value: 'Измеренное значение',
  threshold: 'Допустимое значение',
  name: 'Проверка',
  runs: 'Количество проверок',
  retry: 'Повторить генерацию',
  reasons: 'Причины',
  rejection_reasons: 'Причины отклонения',
  run_id: 'ID запуска',
  event: 'Событие',
  image: 'Файл результата',
  image_metadata: 'Параметры изображения',
  technical_validation_enabled: 'Техническая проверка включена',
  visual_validation_runs: 'Количество визуальных проверок',
  context_images: 'Изображений в контексте',
};

const scalarLabels = {
  passed: 'Пройдено',
  failed: 'Ошибка',
  disabled: 'Отключено',
  accepted: 'Принято',
  rejected: 'Отклонено',
  supported: 'Подтверждено источником',
  inferred: 'Выведено моделью',
  unknown: 'Неизвестно',
  ready_for_context: 'Готово к передаче в контекст',
  visual_validation_unavailable: 'Визуальная проверка недоступна',
};

function readableLabel(key) {
  return fieldLabels[key] || key.replaceAll('_', ' ').replace(/^./, value => value.toUpperCase());
}

function readableValue(value, depth = 0) {
  if (value === null || value === undefined) {
    const empty = document.createElement('span');
    empty.className = 'readable-empty';
    empty.textContent = 'Нет данных';
    return empty;
  }
  if (typeof value !== 'object') {
    const scalar = document.createElement('span');
    scalar.className = `readable-scalar readable-${typeof value}`;
    scalar.textContent = typeof value === 'boolean'
      ? (value ? 'Да' : 'Нет')
      : (scalarLabels[value] || String(value));
    return scalar;
  }
  if (depth >= 5) {
    const compact = document.createElement('pre');
    compact.className = 'readable-compact';
    compact.textContent = JSON.stringify(value, null, 2);
    return compact;
  }
  if (Array.isArray(value)) {
    const list = document.createElement('div');
    list.className = 'readable-list';
    if (!value.length) {
      list.append(readableValue(null, depth + 1));
      return list;
    }
    value.forEach((item, index) => {
      const row = document.createElement('div');
      row.className = 'readable-list-item';
      const marker = document.createElement('span');
      marker.className = 'readable-marker';
      marker.textContent = String(index + 1).padStart(2, '0');
      row.append(marker, readableValue(item, depth + 1));
      list.append(row);
    });
    return list;
  }
  const object = document.createElement('dl');
  object.className = 'readable-object';
  Object.entries(value).forEach(([key, item]) => {
    const row = document.createElement('div');
    row.className = 'readable-row';
    const term = document.createElement('dt');
    term.textContent = readableLabel(key);
    const description = document.createElement('dd');
    description.append(readableValue(item, depth + 1));
    row.append(term, description);
    object.append(row);
  });
  return object;
}

function renderArtifacts(artifacts) {
  const section = $('#handoffs');
  section.classList.toggle('hidden', !artifacts.length);
  const countEnding = artifacts.length % 10 === 1 && artifacts.length % 100 !== 11
    ? 'объект'
    : ([2, 3, 4].includes(artifacts.length % 10) && ![12, 13, 14].includes(artifacts.length % 100) ? 'объекта' : 'объектов');
  $('#artifactCount').textContent = `${artifacts.length} ${countEnding}`;
  const list = $('#artifactList');
  list.replaceChildren();

  artifacts.forEach((artifact, index) => {
    const card = document.createElement('details');
    card.className = 'artifact-card';
    card.open = artifact.path === 'prompt.json' || index === artifacts.length - 1;
    const summary = document.createElement('summary');
    const title = document.createElement('strong');
    title.textContent = artifact.title;
    const flow = document.createElement('span');
    flow.className = 'artifact-flow';
    flow.textContent = `${artifact.producer} → ${artifact.consumer}`;
    summary.append(title, flow);
    card.append(summary);

    const body = document.createElement('div');
    body.className = 'artifact-body';
    const content = artifact.content;
    if (content && typeof content === 'object' && !Array.isArray(content) && ('prompt' in content || 'negative_prompt' in content)) {
      const prompts = document.createElement('div');
      prompts.className = 'prompt-outputs';
      for (const [key, className] of [['prompt', 'positive'], ['negative_prompt', 'negative']]) {
        if (!content[key]) continue;
        const block = document.createElement('section');
        block.className = `prompt-output ${className}`;
        const heading = document.createElement('h3');
        heading.textContent = readableLabel(key);
        const text = document.createElement('p');
        text.textContent = content[key];
        block.append(heading, text);
        prompts.append(block);
      }
      body.append(prompts);
      const remainder = Object.fromEntries(Object.entries(content).filter(([key]) => !['prompt', 'negative_prompt'].includes(key)));
      if (Object.keys(remainder).length) body.append(readableValue(remainder));
    } else {
      body.append(readableValue(content));
    }

    const raw = document.createElement('details');
    raw.className = 'raw-json';
    const rawSummary = document.createElement('summary');
    rawSummary.textContent = 'Показать исходный JSON';
    const rawText = document.createElement('pre');
    rawText.textContent = JSON.stringify(content, null, 2);
    raw.append(rawSummary, rawText);
    body.append(raw);
    card.append(body);
    list.append(card);
  });
}

async function refreshArtifacts(job) {
  if (!job.artifacts_revision || job.artifacts_revision === '0' || job.artifacts_revision === artifactRevision) return;
  const requestedJob = job.id;
  try {
    const data = await api(`/api/jobs/${job.id}/artifacts`);
    if (artifactJobId !== requestedJob) return;
    artifactRevision = data.revision;
    renderArtifacts(data.artifacts || []);
  } catch (error) {
    if (artifactJobId === requestedJob) console.warn('Не удалось обновить данные этапов', error);
  }
}

function setRunning(job) {
  $('#emptyState').classList.add('hidden');
  $('#runState').classList.remove('hidden');
  $('#resultSection').classList.add('hidden');
  $('#downloadResult').classList.add('hidden');
  artifactJobId = job.id;
  artifactRevision = '0';
  $('#handoffs').classList.add('hidden');
  $('#artifactList').replaceChildren();
  updateJob(job);
}

function updateJob(job) {
  const progress = job.status === 'failed' ? job.progress : Math.max(job.progress, 2);
  $('#progressValue').textContent = `${progress}%`;
  $('#progressBar').style.width = `${progress}%`;
  $('#stageLabel').textContent = job.stage_label;
  $('#attemptLabel').textContent = job.attempt ? `Попытка ${job.attempt}` : '';
  $('#logOutput').textContent = (job.logs || []).join('\n') || 'Журнал пока пуст…';
  const status = $('#runStatus');
  status.className = `status-badge ${job.status}`;
  status.textContent = job.status === 'complete' ? 'Завершено' : job.status === 'failed' ? 'Ошибка' : 'Выполняется';
  $('#errorBox').classList.toggle('hidden', !job.error);
  $('#errorBox').textContent = job.error || '';
  refreshArtifacts(job);
  if (job.status === 'complete') showResult(job);
}

function showResult(job) {
  $('#resultSection').classList.remove('hidden');
  $('#resultImage').src = `/api/jobs/${job.id}/image?t=${Date.now()}`;
  $('#downloadResult').href = `/api/jobs/${job.id}/download`;
  $('#downloadResult').classList.remove('hidden');
  const result = job.result || {};
  $('#resultMeta').innerHTML = `<div>${result.status === 'accepted' ? 'Принято валидатором' : 'Требуется ручная проверка'}</div><div>${result.attempts || 1} попытка(и)</div>`;
  const warnings = result.rejection_reasons || [];
  $('#resultWarnings').classList.toggle('hidden', !warnings.length);
  $('#resultWarnings').textContent = warnings.length ? `Обратите внимание: ${warnings.join(' · ')}` : '';
  setTimeout(() => $('#resultSection').scrollIntoView({ behavior: 'smooth', block: 'start' }), 250);
}

async function poll(jobId) {
  clearTimeout(pollTimer);
  try {
    const job = await api(`/api/jobs/${jobId}`);
    updateJob(job);
    if (!['complete', 'failed'].includes(job.status)) pollTimer = setTimeout(() => poll(jobId), 1200);
    else $('#launchButton').disabled = false;
  } catch (error) {
    $('#errorBox').classList.remove('hidden');
    $('#errorBox').textContent = error.message;
    pollTimer = setTimeout(() => poll(jobId), 2500);
  }
}

$('#eventInput').addEventListener('input', event => { $('#charCount').textContent = event.target.value.length; });
$('#resetButton').addEventListener('click', resetForm);
document.addEventListener('keydown', event => {
  if (!event.repeat && event.ctrlKey && event.altKey && event.code === 'KeyD') {
    event.preventDefault();
    if (!stagedModeAvailable) return;
    stagedMode = !stagedMode;
    updateFormDependencies();
  }
});
$('#visualValidationEnabled').addEventListener('change', event => {
  const select = document.querySelector('[data-stage="visual_validator"]');
  if (event.target.checked && select.value === 'unavailable') {
    const available = [...select.options].find(option => option.value !== 'unavailable');
    if (available) {
      select.value = available.value;
      updateDescription(select);
    }
  }
  updateFormDependencies();
});
$('#visualValidationRuns').addEventListener('input', event => {
  event.target.value = Math.min(5, Math.max(1, Number(event.target.value) || 1));
});
$('#dropzone').addEventListener('click', () => $('#imageInput').click());
$('#imageInput').addEventListener('change', event => addImages([...event.target.files]));
$('#dropzone').addEventListener('dragover', event => { event.preventDefault(); if (!$('#dropzone').disabled) $('#dropzone').classList.add('dragging'); });
$('#dropzone').addEventListener('dragleave', () => $('#dropzone').classList.remove('dragging'));
$('#dropzone').addEventListener('drop', event => {
  event.preventDefault();
  $('#dropzone').classList.remove('dragging');
  if (!$('#dropzone').disabled) addImages([...event.dataTransfer.files]);
});
$('#launchForm').addEventListener('submit', async event => {
  event.preventDefault();
  $('#formError').textContent = '';
  $('#launchButton').disabled = true;
  try {
    const splitCategories = value => [...new Set(value.split(',').map(item => item.trim()).filter(Boolean))];
    for (const [index, item] of imageFiles.entries()) {
      if (stagedMode) break;
      if (!splitCategories(item.useFor).length) {
        throw new Error(`Для фотографии №${index + 1} укажите, какие детали использовать.`);
      }
    }
    const images = await Promise.all(imageFiles.map(async item => ({
      name: item.file.name,
      type: item.file.type,
      data: await fileAsBase64(item.file),
      use_for: splitCategories(item.useFor),
      do_not_copy: splitCategories(item.doNotCopy),
    })));
    const job = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        event: $('#eventInput').value,
        demo: stagedMode,
        providers: selections(),
        images,
        credentials: {
          kaggle_username: $('#kaggleUsername').value,
          kaggle_token: $('#kaggleToken').value,
          hf_token: $('#hfToken').value,
          gpt_token: $('#gptToken').value,
        },
        visual_validation: {
          enabled: $('#visualValidationEnabled').checked,
          runs: Number($('#visualValidationRuns').value),
        },
        technical_validation: {
          enabled: $('#technicalValidationEnabled').checked,
        },
      }),
    });
    $('#kaggleToken').value = '';
    $('#hfToken').value = '';
    $('#gptToken').value = '';
    setRunning(job);
    poll(job.id);
  } catch (error) {
    $('#formError').textContent = error.message;
    $('#launchButton').disabled = false;
  }
});

api('/api/options').then(renderStages).catch(error => { $('#stageList').innerHTML = `<div class="error-box">${error.message}</div>`; });
