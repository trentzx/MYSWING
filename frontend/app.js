const $ = (selector) => document.querySelector(selector);
const player = $('#swing-video');
let videoUrl;
let report;
let reportName;
let currentVideoName;
const phaseButtons = [...document.querySelectorAll('[data-phase]')];

function updatePhases() {
  const matched = report && currentVideoName && [report.input, report.output_video].some(path => typeof path === 'string' && path.split(/[\\/]/).pop() === currentVideoName);
  phaseButtons.forEach(button => {
    const time = report ? report.phases[button.dataset.phase] / report.video.fps : NaN;
    button.disabled = !(matched && Number.isFinite(player.duration) && time <= player.duration);
    button.querySelector('.phase-time').textContent = Number.isFinite(time) ? `${time.toFixed(2)}s` : '—';
  });
  if (report) $('#report-status').textContent = report.swing_location?.reliable === false
    ? 'Low confidence: this recording may not contain a clean, continuous swing. Interpret these metrics cautiously.'
    : report.swing_location?.reliable !== true ? 'This report does not include a recording-confidence assessment.' : 'Report loaded.';
  if (report && currentVideoName && !matched) $('#report-status').textContent += ' Choose the video named in this report to enable phase navigation.';
}

function loadVideo(file) {
  if (!file) return;
  if (!/\.(mp4|mov|webm)$/i.test(file.name) || file.size > 250 * 1024 * 1024) {
    $('#video-status').textContent = 'Choose an MP4, MOV or WebM video smaller than 250 MB.';
    return;
  }
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  videoUrl = URL.createObjectURL(file);
  currentVideoName = file.name;
  player.src = videoUrl;
  player.hidden = false;
  $('#empty-video').hidden = true;
  $('#replace-video').hidden = false;
  $('#empty-duration').hidden = true;
  $('#video-name').textContent = file.name;
  $('#video-tag').textContent = 'LOCAL VIDEO';
  $('#video-status').textContent = 'Video ready for review. Import an analysis report to see measured results; choosing a video does not run analysis.';
  updatePhases();
}

for (const id of ['#choose-video', '#replace-video']) $(id).addEventListener('click', () => $('#video-input').click());
$('#video-input').addEventListener('change', event => { loadVideo(event.target.files[0]); event.target.value = ''; });
$('#choose-report').addEventListener('click', () => $('#report-input').click());
player.addEventListener('loadedmetadata', updatePhases);
player.addEventListener('error', () => {
  $('#video-status').textContent = 'This browser cannot play this video. Try an H.264 MP4 or a WebM file.';
  phaseButtons.forEach(button => { button.disabled = true; });
});
phaseButtons.forEach(button => button.addEventListener('click', () => {
  if (!report || button.disabled) return;
  player.pause();
  player.currentTime = report.phases[button.dataset.phase] / report.video.fps;
}));

const zone = $('#drop-zone');
for (const event of ['dragenter', 'dragover']) zone.addEventListener(event, e => { e.preventDefault(); zone.classList.add('dragging'); });
for (const event of ['dragleave', 'drop']) zone.addEventListener(event, e => { e.preventDefault(); zone.classList.remove('dragging'); });
zone.addEventListener('drop', event => loadVideo(event.dataTransfer.files[0]));

$('#report-input').addEventListener('change', async event => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file) return;
  try {
    if (file.size > 1024 * 1024) throw new Error('Report exceeds 1 MB.');
    const data = JSON.parse(await file.text());
    const m = data.metrics;
    if (!m || !['tempo_ratio', 'head_stability_pct', 'spine_tilt_deg'].every(key => Number.isFinite(m[key]) && m[key] >= 0) ||
      !Number.isFinite(data.video?.fps) || data.video.fps <= 0 ||
      !['address', 'top', 'impact'].every(key => Number.isInteger(data.phases?.[key]) && data.phases[key] >= 0) ||
      !(data.phases.address < data.phases.top && data.phases.top < data.phases.impact)) throw new Error('Invalid report.');
    report = data;
    reportName = file.name;
    $('#tempo').replaceChildren(document.createTextNode(m.tempo_ratio.toFixed(1)), Object.assign(document.createElement('small'), { textContent: ' : 1' }));
    $('#head').replaceChildren(document.createTextNode(m.head_stability_pct.toFixed(1)), Object.assign(document.createElement('small'), { textContent: '%' }));
    $('#spine').replaceChildren(document.createTextNode(m.spine_tilt_deg.toFixed(1)), Object.assign(document.createElement('small'), { textContent: '°' }));
    $('#tempo-bar').style.width = `${100 * (m.tempo_ratio / (m.tempo_ratio + 1))}%`;
    $('#report-tag').textContent = 'REPORT LOADED';
    $('#report-tag').title = reportName;
    $('#choose-report').textContent = 'Replace analysis report ↗';
    updatePhases();
  } catch {
    $('#report-status').textContent = 'Could not read this report. Choose a Golf Swing Analyzer JSON file under 1 MB with valid metrics, frame rate, and swing phases. Your previous report is unchanged.';
  }
});
window.addEventListener('pagehide', () => { if (videoUrl) URL.revokeObjectURL(videoUrl); });
