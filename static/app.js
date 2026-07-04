let performanceMetrics = null;
let activeModel = 'random_forest';
let charts = {};

// Circle Progress Ring Config
const circle = document.querySelector('.progress-ring-fill');
const radius = circle.r.baseVal.value;
const circumference = radius * 2 * Math.PI;
circle.style.strokeDasharray = `${circumference} ${circumference}`;
circle.style.strokeDashoffset = circumference;

function setProgress(percent) {
    const offset = circumference - (percent / 100 * circumference);
    circle.style.strokeDashoffset = offset;
}

document.addEventListener('DOMContentLoaded', () => {
    setProgress(0);
    initApp();
    setupEventListeners();
});

async function initApp() {
    await fetchPerformance();
    triggerPrediction();
}

function setupEventListeners() {
    // 1. Model selection
    const algoSelect = document.getElementById('algorithm-select');
    algoSelect.addEventListener('change', (e) => {
        activeModel = e.target.value;
        triggerPrediction();
        showToast(`Switched active model to ${e.target.options[e.target.selectedIndex].text}`);
    });

    // 2. Sliders listeners
    const sliderIds = ['age', 'history', 'income', 'debt', 'savings', 'missed', 'limit', 'balance'];
    sliderIds.forEach(id => {
        const input = document.getElementById(`input-${id}`);
        const display = document.getElementById(`val-${id}`);
        
        if (input && display) {
            input.addEventListener('input', () => {
                let val = parseFloat(input.value);
                
                // Update label display
                if (id === 'history') {
                    display.innerText = `${val} yrs`;
                } else if (['income', 'debt', 'savings', 'limit', 'balance'].includes(id)) {
                    display.innerText = `$${val.toLocaleString()}`;
                } else {
                    display.innerText = val;
                }
                
                // Keep balance slider max synced with limit
                if (id === 'limit') {
                    const balanceSlider = document.getElementById('input-balance');
                    balanceSlider.max = val;
                    if (parseFloat(balanceSlider.value) > val) {
                        balanceSlider.value = val;
                        document.getElementById('val-balance').innerText = `$${val.toLocaleString()}`;
                    }
                }
                
                // Recalculate features and predictions
                recalculateRatios();
                triggerPrediction();
            });
        }
    });
}

function recalculateRatios() {
    const income = parseFloat(document.getElementById('input-income').value) || 0;
    const debt = parseFloat(document.getElementById('input-debt').value) || 0;
    const savings = parseFloat(document.getElementById('input-savings').value) || 0;
    const limit = parseFloat(document.getElementById('input-limit').value) || 1;
    const balance = parseFloat(document.getElementById('input-balance').value) || 0;

    const dti = income > 0 ? (debt / income) * 100 : 0;
    const utilization = limit > 0 ? (balance / limit) * 100 : 0;
    const savingsRatio = income > 0 ? (savings / income) * 100 : 0;

    document.getElementById('ratio-dti').innerText = `${dti.toFixed(1)}%`;
    document.getElementById('ratio-util').innerText = `${utilization.toFixed(1)}%`;
    document.getElementById('ratio-savings').innerText = `${savingsRatio.toFixed(1)}%`;
}

// Fetch performance stats on startup
async function fetchPerformance() {
    try {
        const response = await fetch('/api/performance');
        if (!response.ok) throw new Error('Metrics loading failed');
        
        performanceMetrics = await response.json();
        renderCharts();
    } catch (err) {
        console.error(err);
        showToast('Error loading performance graphs');
    }
}

// Render metrics comparisons
function renderCharts() {
    const models = Object.keys(performanceMetrics);
    const labelMapping = {
        'logistic_regression': 'Logistic Reg.',
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest'
    };
    
    const displayLabels = models.map(m => labelMapping[m] || m);
    
    // --- 1. Metrics comparison bar chart ---
    const ctxMetrics = document.getElementById('chart-metrics').getContext('2d');
    if (charts.metrics) charts.metrics.destroy();
    
    charts.metrics = new Chart(ctxMetrics, {
        type: 'bar',
        data: {
            labels: displayLabels,
            datasets: [
                { label: 'Accuracy', data: models.map(m => performanceMetrics[m].accuracy), backgroundColor: '#3b82f6' },
                { label: 'Precision', data: models.map(m => performanceMetrics[m].precision), backgroundColor: '#10b981' },
                { label: 'Recall', data: models.map(m => performanceMetrics[m].recall), backgroundColor: '#f59e0b' },
                { label: 'F1-Score', data: models.map(m => performanceMetrics[m].f1_score), backgroundColor: '#ef4444' },
                { label: 'ROC-AUC', data: models.map(m => performanceMetrics[m].roc_auc), backgroundColor: '#8b5cf6' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', boxWidth: 12, font: { size: 10 } } }
            },
            scales: {
                x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } },
                y: { min: 0.5, max: 1.0, ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } }
            }
        }
    });

    // --- 2. ROC curves comparison ---
    const ctxRoc = document.getElementById('chart-roc').getContext('2d');
    if (charts.roc) charts.roc.destroy();
    
    const colors = {
        'random_forest': '#3b82f6',
        'decision_tree': '#8b5cf6',
        'logistic_regression': '#10b981'
    };
    
    const datasets = models.map(m => {
        const points = performanceMetrics[m].roc_curve.map(p => ({ x: p.fpr, y: p.tpr }));
        return {
            label: labelMapping[m] || m,
            data: points,
            borderColor: colors[m] || '#fff',
            borderWidth: 2,
            showLine: true,
            fill: false,
            pointRadius: 0
        };
    });
    
    // Baseline
    datasets.push({
        label: 'Baseline',
        data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
        borderColor: 'rgba(255, 255, 255, 0.15)',
        borderWidth: 1,
        borderDash: [5, 5],
        showLine: true,
        fill: false,
        pointRadius: 0
    });

    charts.roc = new Chart(ctxRoc, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', boxWidth: 12, font: { size: 10 } } }
            },
            scales: {
                x: { type: 'linear', position: 'bottom', ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' }, min: 0, max: 1 },
                y: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' }, min: 0, max: 1 }
            }
        }
    });
}

// Trigger prediction
async function triggerPrediction() {
    try {
        const payload = {
            age: parseInt(document.getElementById('input-age').value),
            annual_income: parseFloat(document.getElementById('input-income').value),
            total_debt: parseFloat(document.getElementById('input-debt').value),
            missed_payments: parseInt(document.getElementById('input-missed').value),
            savings: parseFloat(document.getElementById('input-savings').value),
            credit_card_limit: parseFloat(document.getElementById('input-limit').value),
            credit_card_balance: parseFloat(document.getElementById('input-balance').value),
            credit_history_length_years: parseInt(document.getElementById('input-history').value),
            model_name: activeModel
        };

        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('Prediction API failure');
        const res = await response.json();
        
        // Update dashboard score gauge and labels
        updateUI(res);
        
    } catch (err) {
        console.error(err);
    }
}

function updateUI(res) {
    const isApproved = res.decision === 'Approved';
    
    // Set score circle gauge
    document.getElementById('result-score').innerText = res.score;
    const percent = ((res.score - 300) / 550) * 100;
    setProgress(percent);
    
    // Update stroke color
    circle.style.stroke = isApproved ? 'var(--success-color)' : 'var(--danger-color)';
    
    // Risk badge
    const badge = document.getElementById('result-risk-badge');
    badge.innerText = `${res.risk_category} RISK`;
    badge.className = `risk-badge ${res.risk_category.replace(' ', '_')}`;
    
    // Status text
    const decEl = document.getElementById('result-decision');
    decEl.innerText = res.decision.toUpperCase();
    decEl.className = isApproved ? 'text-success' : 'text-danger';
    
    // Probability of default
    document.getElementById('result-pd').innerText = `${(res.probability_of_default * 100).toFixed(1)}%`;
}

// Toast helper
function showToast(message) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<span>${message}</span>`;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => { container.removeChild(toast); }, 300);
    }, 3000);
}
