async function predecir() {
  const btn = document.getElementById('predictBtn');
  const result = document.getElementById('result');
  const texto = document.getElementById('inputText').value.trim();
  if (!texto) { result.className = 'error show'; result.innerHTML = '\u26A0\uFE0F Ingresa un texto.'; return; }
  btn.disabled = true; btn.innerHTML = '<span class="loader"></span> Clasificando...';
  result.className = '';
  try {
    const res = await fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({texto}) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Error');
    let bars = '';
    const colors = ['#3b82f6','#8b5cf6','#ef4444','#10b981','#f59e0b'];
    const total = data.top_n.reduce((a,b) => a + b[1], 0);
    data.top_n.forEach((p, i) => {
      const pct = ((p[1] / total) * 100).toFixed(1);
      bars += '<div class="prob-seg" style="width:' + pct + '%;background:' + colors[i % colors.length] + '">' + p[0] + ' ' + pct + '%</div>';
    });
    result.className = 'pred show';
    result.innerHTML = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem">' +
      '<strong>Categor\u00EDa: <span style="color:#34d399">' + data.categoria_predicha + '</span></strong>' +
      '<span>Confianza: <strong>' + data.confianza + '%</strong></span></div>' +
      '<div class="prob-bar">' + bars + '</div>';
  } catch(e) {
    result.className = 'error show';
    result.innerHTML = '\u26A0\uFE0F ' + e.message;
  } finally {
    btn.disabled = false; btn.innerHTML = 'Clasificar';
  }
}

function colorizeConfusionMatrix() {
  const table = document.querySelector('.cm-table');
  if (!table) return;
  const rows = table.querySelectorAll('tbody tr');
  let maxVal = 0;
  rows.forEach(row => {
    row.querySelectorAll('td').forEach(td => {
      const v = parseInt(td.textContent, 10);
      if (!isNaN(v) && v > maxVal) maxVal = v;
    });
  });
  if (maxVal === 0) return;
  rows.forEach(row => {
    row.querySelectorAll('td').forEach(td => {
      const v = parseInt(td.textContent, 10);
      if (isNaN(v)) return;
      const ratio = v / maxVal;
      const r = Math.round(255 * (1 - ratio));
      const g = Math.round(255 * (1 - ratio));
      const b = 255;
      const textCol = ratio < 0.5 ? '#1e293b' : '#f0f9ff';
      td.style.backgroundColor = 'rgb(' + r + ',' + g + ',' + b + ')';
      td.style.color = textCol;
    });
  });
}

colorizeConfusionMatrix();