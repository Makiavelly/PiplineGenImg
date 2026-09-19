const $ = (selector) => document.querySelector(selector);
let options = null;
let pollTimer = null;
let imageFiles = [];

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
  document.querySelectorAll('[data-stage]').forEach(select => {
    select.addEventListener('change', () => { updateDescription(select); updateFormDependencies(); });
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

function updateFormDependencies() {
  const values = Object.values(selections());
  const usesKaggle = values.some(value => value.startsWith('kaggle'));
  $('#credentials').classList.toggle('hidden', !usesKaggle);
  $('#hfToken').closest('label').classList.toggle('hidden', selections().image_generator !== 'kaggle_sd35');
  const provider = selectedImageProvider();
  const supportsImages = Boolean(provider && provider.supports_images);
  $('#imageInput').disabled = !supportsImages;
  $('#dropzone').disabled = !supportsImages;
  $('#uploadCapability').textContent = supportsImages
    ? 'Изображения будут переданы непосредственно генератору'
    : 'Выбранная модель принимает только текст';
  if (!supportsImages && imageFiles.length) clearImages();
}

function resetForm() {
  if (!options) return;
  for (const stage of options.stages) {
    const select = document.querySelector(`[data-stage="${stage.id}"]`);
    select.value = stage.selected;
    updateDescription(select);
  }
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
      <img src="${item.url}" alt="">
      <div><strong>${escapeHtml(item.file.name)}</strong><small>${formatBytes(item.file.size)}</small></div>
      <button type="button" data-remove-image="${index}" aria-label="Удалить фотографию">×</button>
    </div>
  `).join('');
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
  if (!selectedImageProvider()?.supports_images) {
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
    imageFiles.push({ file, url: URL.createObjectURL(file) });
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

function setRunning(job) {
  $('#emptyState').classList.add('hidden');
  $('#runState').classList.remove('hidden');
  $('#runId').textContent = `#${job.id}`;
  $('#resultSection').classList.add('hidden');
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
  if (job.status === 'complete') showResult(job);
}

function showResult(job) {
  $('#resultSection').classList.remove('hidden');
  $('#resultImage').src = `/api/jobs/${job.id}/image?t=${Date.now()}`;
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
    const images = await Promise.all(imageFiles.map(async item => ({
      name: item.file.name,
      type: item.file.type,
      data: await fileAsBase64(item.file),
    })));
    const job = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        event: $('#eventInput').value,
        providers: selections(),
        images,
        credentials: {
          kaggle_username: $('#kaggleUsername').value,
          kaggle_token: $('#kaggleToken').value,
          hf_token: $('#hfToken').value,
        },
      }),
    });
    $('#kaggleToken').value = '';
    $('#hfToken').value = '';
    setRunning(job);
    poll(job.id);
  } catch (error) {
    $('#formError').textContent = error.message;
    $('#launchButton').disabled = false;
  }
});

api('/api/options').then(renderStages).catch(error => { $('#stageList').innerHTML = `<div class="error-box">${error.message}</div>`; });
