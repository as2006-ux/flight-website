const $ = id => document.getElementById(id);
function table(rows, heads) {
  const keys = Object.keys(heads);
  return '<div class="tw"><table><thead><tr>' + keys.map(k => `<th>${heads[k]}</th>`).join('') +
    '</tr></thead><tbody>' + rows.map(r => '<tr>' + keys.map(k => `<td>${r[k]}</td>`).join('') + '</tr>').join('') +
    '</tbody></table></div>';
}
const matrix = (m, rowLabel, cols) => table(m.map((r, i) => ({ n: `${rowLabel} ${i}`, ...Object.fromEntries(r.map((v, j) => ['c' + j, v])) })),
  { n: '', ...Object.fromEntries(cols.map((c, j) => ['c' + j, c])) });

async function load() {
  try {
    const r = await (await fetch('/api/results')).json();
    $('reg').innerHTML = '<h3>Regression</h3>' + table(r.regression, { model: 'Model', mae: 'MAE (min)', rmse: 'RMSE (min)', r2: 'R²' });
    $('cls').innerHTML = '<h3>Classification (delay &gt; 15 min)</h3>' + table(r.classification,
      { model: 'Model', accuracy: 'Accuracy %', precision: 'Precision %', recall: 'Recall %', f1: 'F1 %' }) +
      '<p class="small">Accuracy is misleading here: about 82% of flights are on time.</p>';
    $('km').innerHTML = '<h3>K-Means</h3>' + table(r.kmeans, { k: 'K', inertia: 'Inertia', silhouette: 'Silhouette' });
    const g = r.gmm;
    $('gmm').textContent = `GMM (K=${g.k}): silhouette ${g.silhouette}, log-likelihood ${g.log_likelihood}, ${g.iterations} EM iterations, weights ${g.weights.join('%, ')}%.`;
    $('pca').innerHTML = '<h3>PCA variance explained (%)</h3>' + table([Object.fromEntries(r.pca_variance.map((v, i) => ['p' + i, v]))],
      Object.fromEntries(r.pca_variance.map((_, i) => ['p' + i, 'PC' + (i + 1)])));
    const h = r.hmm;
    $('hmm').textContent = `${h.days} daily observations (${h.train} train, ${h.test} test). Markov next-day accuracy ${h.markov_accuracy}%. Held-out log-likelihood ${h.test_loglik} (${h.test_loglik_per_day} per day). ${h.baum_welch_iterations} Baum–Welch iterations.`;
    $('tr').innerHTML = '<h3>Transition matrix</h3>' + matrix(h.transition, 'State', ['→ 0', '→ 1', '→ 2']);
    $('em').innerHTML = '<h3>Emission matrix</h3>' + matrix(h.emission, 'State', ['Low day', 'Medium day', 'High day']);
    $('lim').innerHTML = r.limitations.map(l => `<li>${l}</li>`).join('');
  } catch (e) {
    $('reg').innerHTML = '<p class="err">Could not load results. Is the backend running?</p>';
  }
}

$('f').addEventListener('submit', async e => {
  e.preventDefault();
  const body = Object.fromEntries(new FormData(e.target));
  $('out').className = ''; $('out').textContent = 'Estimating…';
  try {
    const res = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const d = await res.json();
    if (!res.ok) { $('out').className = 'err'; $('out').textContent = d.error; return; }
    $('out').innerHTML = `Estimated arrival delay: <b>${d.predicted_arrival_delay_minutes} min</b><br><span class="small">${d.caveat}</span>`;
  } catch (err) { $('out').className = 'err'; $('out').textContent = 'Request failed. Check that the backend is running.'; }
});
load();
