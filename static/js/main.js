// static/js/main.js
document.getElementById('predictBtn').addEventListener('click', async () => {
  const ticker = document.getElementById('ticker').value || 'BTC-USD';
  const days = document.getElementById('days').value || 365;
  const status = document.getElementById('status');
  status.textContent = 'Requesting prediction...';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker, history_days: days })
    });
    const data = await res.json();
    if (res.ok) {
      document.getElementById('history').textContent = JSON.stringify(data.history.close.slice(-20), null, 2);
      document.getElementById('preds').textContent = JSON.stringify(data.predictions, null, 2);
      status.textContent = 'Done';
    } else {
      status.textContent = 'Error: ' + (data.error || JSON.stringify(data));
    }
  } catch (err) {
    status.textContent = 'Network error: ' + err.message;
  }
});
