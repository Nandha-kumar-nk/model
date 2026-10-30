// ==============================================================================
// SMART GARBAGE CLASSIFICATION SYSTEM - DASHBOARD CONTROLLER (dashboard.js)
// ==============================================================================

// Baseline demo data (used on fresh launch or demo reset)
const DEFAULT_BASELINE = {
    capacities: {
        organic: 65,
        recyclable: 42,
        nonRecyclable: 78,
        hazardous: 25
    },
    baseItemCounts: {
        organic: 24,
        recyclable: 31,
        nonRecyclable: 38,
        hazardous: 7
    },
    sampleHistory: [
        {
            id: 'demo_1',
            timestamp: new Date(Date.now() - 3600000 * 2).toISOString(),
            timeString: "10:32 AM",
            itemName: "Banana Peel",
            category: "Organic",
            confidence: 95,
            bin: "Organic Bin"
        },
        {
            id: 'demo_2',
            timestamp: new Date(Date.now() - 3600000 * 1.5).toISOString(),
            timeString: "10:35 AM",
            itemName: "Water Bottle",
            category: "Recyclable",
            confidence: 92,
            bin: "Recyclable Bin"
        },
        {
            id: 'demo_3',
            timestamp: new Date(Date.now() - 3600000 * 1).toISOString(),
            timeString: "11:10 AM",
            itemName: "Contaminated Snack Wrapper",
            category: "Non-Recyclable",
            confidence: 94,
            bin: "Non-Recyclable Bin"
        },
        {
            id: 'demo_4',
            timestamp: new Date(Date.now() - 3600000 * 0.5).toISOString(),
            timeString: "11:45 AM",
            itemName: "Dry Cell Battery",
            category: "Hazardous",
            confidence: 98,
            bin: "Hazardous Bin"
        },
        {
            id: 'demo_5',
            timestamp: new Date(Date.now() - 3600000 * 0.2).toISOString(),
            timeString: "12:15 PM",
            itemName: "Tomato & Vegetable Scraps",
            category: "Organic",
            confidence: 96,
            bin: "Organic Bin"
        }
    ]
};

document.addEventListener('DOMContentLoaded', () => {
    // Navigation toggle
    const mobileToggle = document.getElementById('mobileToggle');
    const navLinks = document.getElementById('navLinks');
    if (mobileToggle && navLinks) {
        mobileToggle.addEventListener('click', () => {
            navLinks.classList.toggle('mobile-open');
        });
    }

    // Initialize Dashboard
    initializeDashboard();

    // Event Listeners for Dashboard Buttons
    const btnSimulateAlert = document.getElementById('btnSimulateAlert');
    const btnResetDemo = document.getElementById('btnResetDemo');
    const btnClearHistory = document.getElementById('btnClearHistory');

    if (btnSimulateAlert) {
        btnSimulateAlert.addEventListener('click', () => {
            simulate90PercentAlert();
        });
    }

    if (btnResetDemo) {
        btnResetDemo.addEventListener('click', () => {
            resetDemoDashboard();
        });
    }

    if (btnClearHistory) {
        btnClearHistory.addEventListener('click', () => {
            clearClassificationHistory();
        });
    }
});

/**
 * Initialize or synchronize state from localStorage
 */
function initializeDashboard() {
    ensureInitialLocalStorage();
    renderBinCapacityCards();
    renderStatisticsSummary();
    renderHistoryTable();
    checkAndDisplayCapacityAlerts();
}

/**
 * Ensure sensible demo data exists in localStorage
 */
function ensureInitialLocalStorage() {
    try {
        if (!localStorage.getItem('binCapacities')) {
            localStorage.setItem('binCapacities', JSON.stringify(DEFAULT_BASELINE.capacities));
        }

        if (!localStorage.getItem('baseCounts')) {
            localStorage.setItem('baseCounts', JSON.stringify(DEFAULT_BASELINE.baseItemCounts));
        }

        if (!localStorage.getItem('classificationHistory')) {
            localStorage.setItem('classificationHistory', JSON.stringify(DEFAULT_BASELINE.sampleHistory));
        }
    } catch (e) {
        console.warn("LocalStorage access issue:", e);
    }
}

/**
 * Fetch and calculate aggregated waste counts
 */
function calculateAggregatedStats() {
    let baseCounts = DEFAULT_BASELINE.baseItemCounts;
    try {
        const storedBase = localStorage.getItem('baseCounts');
        if (storedBase) baseCounts = JSON.parse(storedBase);
    } catch (e) {}

    let history = [];
    try {
        const storedHistory = localStorage.getItem('classificationHistory');
        if (storedHistory) history = JSON.parse(storedHistory);
    } catch (e) {}

    // Count recent dynamically added classifications (beyond default baseline)
    const recentCounts = {
        organic: 0,
        recyclable: 0,
        nonRecyclable: 0,
        hazardous: 0
    };

    history.forEach(item => {
        // If not a default seeded item ID, count it
        if (!item.id || !item.id.startsWith('demo_')) {
            const cat = (item.category || '').toLowerCase();
            if (cat.includes('organic')) recentCounts.organic++;
            else if (cat.includes('recyclable') && !cat.includes('non')) recentCounts.recyclable++;
            else if (cat.includes('non')) recentCounts.nonRecyclable++;
            else if (cat.includes('hazardous')) recentCounts.hazardous++;
        }
    });

    const totalOrganic = baseCounts.organic + recentCounts.organic;
    const totalRecyclable = baseCounts.recyclable + recentCounts.recyclable;
    const totalNonRecyclable = baseCounts.nonRecyclable + recentCounts.nonRecyclable;
    const totalHazardous = baseCounts.hazardous + recentCounts.hazardous;
    const grandTotal = totalOrganic + totalRecyclable + totalNonRecyclable + totalHazardous;

    return {
        organic: totalOrganic,
        recyclable: totalRecyclable,
        nonRecyclable: totalNonRecyclable,
        hazardous: totalHazardous,
        grandTotal: grandTotal
    };
}

/**
 * Render the 4 Bin Capacity Cards (Organic, Recyclable, Non-Recyclable, Hazardous)
 */
function renderBinCapacityCards() {
    let capacities = DEFAULT_BASELINE.capacities;
    try {
        const storedCapacities = localStorage.getItem('binCapacities');
        if (storedCapacities) capacities = JSON.parse(storedCapacities);
    } catch (e) {}

    const stats = calculateAggregatedStats();

    // Map each category
    const categories = [
        { key: 'organic', name: 'Organic', barId: 'barOrganic', percentId: 'percentOrganic', countId: 'countOrganic', cardId: 'cardOrganic' },
        { key: 'recyclable', name: 'Recyclable', barId: 'barRecyclable', percentId: 'percentRecyclable', countId: 'countRecyclable', cardId: 'cardRecyclable' },
        { key: 'nonRecyclable', name: 'Non-Recyclable', barId: 'barNonRecyclable', percentId: 'percentNonRecyclable', countId: 'countNonRecyclable', cardId: 'cardNonRecyclable' },
        { key: 'hazardous', name: 'Hazardous', barId: 'barHazardous', percentId: 'percentHazardous', countId: 'countHazardous', cardId: 'cardHazardous' }
    ];

    categories.forEach(cat => {
        const capacityVal = Math.min(100, Math.max(0, capacities[cat.key] || 0));
        const itemCount = stats[cat.key] || 0;

        const percentEl = document.getElementById(cat.percentId);
        const countEl = document.getElementById(cat.countId);
        const barEl = document.getElementById(cat.barId);
        const cardEl = document.getElementById(cat.cardId);

        if (percentEl) percentEl.textContent = `${capacityVal}%`;
        if (countEl) countEl.textContent = `${itemCount} items`;

        if (barEl) {
            barEl.style.width = `${capacityVal}%`;
            if (capacityVal >= 90) {
                barEl.classList.add('progress-critical');
            } else {
                barEl.classList.remove('progress-critical');
            }
        }

        if (cardEl) {
            if (capacityVal >= 90) {
                cardEl.classList.add('alert-active');
            } else {
                cardEl.classList.remove('alert-active');
            }
        }
    });
}

/**
 * Render Summary Statistics (Totals and Percentages) & Distribution Chart
 */
function renderStatisticsSummary() {
    const stats = calculateAggregatedStats();

    const totalClassifiedEl = document.getElementById('totalClassified');
    const organicShareEl = document.getElementById('organicShare');
    const recyclableShareEl = document.getElementById('recyclableShare');
    const nonRecyclableShareEl = document.getElementById('nonRecyclableShare');
    const hazardousShareEl = document.getElementById('hazardousShare');

    if (totalClassifiedEl) totalClassifiedEl.textContent = stats.grandTotal;

    const calcPct = (count) => stats.grandTotal > 0 ? Math.round((count / stats.grandTotal) * 100) : 0;
    const orgPct = calcPct(stats.organic);
    const recPct = calcPct(stats.recyclable);
    const nonPct = calcPct(stats.nonRecyclable);
    const hazPct = calcPct(stats.hazardous);

    if (organicShareEl) organicShareEl.textContent = `${orgPct}%`;
    if (recyclableShareEl) recyclableShareEl.textContent = `${recPct}%`;
    if (nonRecyclableShareEl) nonRecyclableShareEl.textContent = `${nonPct}%`;
    if (hazardousShareEl) hazardousShareEl.textContent = `${hazPct}%`;

    // Dynamic Stacked Chart Bar Widths
    const chartBarOrganic = document.getElementById('chartBarOrganic');
    const chartBarRecyclable = document.getElementById('chartBarRecyclable');
    const chartBarNonRecyclable = document.getElementById('chartBarNonRecyclable');
    const chartBarHazardous = document.getElementById('chartBarHazardous');

    if (chartBarOrganic) {
        chartBarOrganic.style.width = `${orgPct}%`;
        chartBarOrganic.title = `Organic: ${orgPct}% (${stats.organic} items)`;
    }
    if (chartBarRecyclable) {
        chartBarRecyclable.style.width = `${recPct}%`;
        chartBarRecyclable.title = `Recyclable: ${recPct}% (${stats.recyclable} items)`;
    }
    if (chartBarNonRecyclable) {
        chartBarNonRecyclable.style.width = `${nonPct}%`;
        chartBarNonRecyclable.title = `Non-Recyclable: ${nonPct}% (${stats.nonRecyclable} items)`;
    }
    if (chartBarHazardous) {
        chartBarHazardous.style.width = `${hazPct}%`;
        chartBarHazardous.title = `Hazardous: ${hazPct}% (${stats.hazardous} items)`;
    }

    // Chart Legend Text
    const chartValOrganic = document.getElementById('chartValOrganic');
    const chartValRecyclable = document.getElementById('chartValRecyclable');
    const chartValNonRecyclable = document.getElementById('chartValNonRecyclable');
    const chartValHazardous = document.getElementById('chartValHazardous');

    if (chartValOrganic) chartValOrganic.textContent = `${orgPct}% (${stats.organic} items)`;
    if (chartValRecyclable) chartValRecyclable.textContent = `${recPct}% (${stats.recyclable} items)`;
    if (chartValNonRecyclable) chartValNonRecyclable.textContent = `${nonPct}% (${stats.nonRecyclable} items)`;
    if (chartValHazardous) chartValHazardous.textContent = `${hazPct}% (${stats.hazardous} items)`;
}

/**
 * Render Classification History Table
 */
function renderHistoryTable() {
    const tableBody = document.getElementById('historyTableBody');
    const emptyPlaceholder = document.getElementById('emptyHistoryPlaceholder');
    if (!tableBody) return;

    let history = [];
    try {
        const storedHistory = localStorage.getItem('classificationHistory');
        if (storedHistory) history = JSON.parse(storedHistory);
    } catch (e) {}

    tableBody.innerHTML = '';

    if (!history || history.length === 0) {
        if (emptyPlaceholder) emptyPlaceholder.style.display = 'block';
        return;
    }

    if (emptyPlaceholder) emptyPlaceholder.style.display = 'none';

    history.forEach(item => {
        const tr = document.createElement('tr');

        // Badge class based on category
        let badgeClass = 'badge-organic';
        const cat = (item.category || '').toLowerCase();
        if (cat.includes('recyclable') && !cat.includes('non')) badgeClass = 'badge-recyclable';
        else if (cat.includes('non')) badgeClass = 'badge-non-recyclable';
        else if (cat.includes('hazardous')) badgeClass = 'badge-hazardous';

        tr.innerHTML = `
            <td class="time-cell">${escapeHtml(item.timeString || 'Recently')}</td>
            <td><strong>${escapeHtml(item.itemName || 'Waste Sample')}</strong></td>
            <td><span class="badge ${badgeClass}">${escapeHtml(item.category)}</span></td>
            <td><span style="font-family: var(--font-mono); color: var(--accent-cyan);">${escapeHtml(item.confidence.toString())}%</span></td>
            <td>${escapeHtml(item.bin)}</td>
        `;
        tableBody.appendChild(tr);
    });
}

/**
 * Check if any bin is >= 90% and display collection alert banner
 */
function checkAndDisplayCapacityAlerts() {
    let capacities = DEFAULT_BASELINE.capacities;
    try {
        const storedCapacities = localStorage.getItem('binCapacities');
        if (storedCapacities) capacities = JSON.parse(storedCapacities);
    } catch (e) {}

    const alertBanner = document.getElementById('collectionAlertBanner');
    const alertMessage = document.getElementById('alertMessageText');
    const alertSubtext = document.getElementById('alertSubtext');

    // Find any bins >= 90%
    const criticalBins = [];
    if (capacities.organic >= 90) criticalBins.push({ name: 'ORGANIC', val: capacities.organic });
    if (capacities.recyclable >= 90) criticalBins.push({ name: 'RECYCLABLE', val: capacities.recyclable });
    if (capacities.nonRecyclable >= 90) criticalBins.push({ name: 'NON-RECYCLABLE', val: capacities.nonRecyclable });
    if (capacities.hazardous >= 90) criticalBins.push({ name: 'HAZARDOUS', val: capacities.hazardous });

    if (criticalBins.length > 0 && alertBanner && alertMessage) {
        const primaryAlert = criticalBins[0];
        alertMessage.textContent = `${primaryAlert.name} BIN HAS REACHED ${primaryAlert.val}% CAPACITY.`;
        if (alertSubtext) {
            alertSubtext.textContent = criticalBins.length > 1 
                ? `Collection required immediately for ${criticalBins.map(b => b.name).join(', ')} bins.`
                : "Collection required.";
        }
        alertBanner.classList.add('active');
    } else if (alertBanner) {
        alertBanner.classList.remove('active');
    }
}

/**
 * Simulate 90% Capacity Alert (for live grading / demonstration)
 */
function simulate90PercentAlert() {
    try {
        const capacitiesJson = localStorage.getItem('binCapacities');
        const capacities = capacitiesJson ? JSON.parse(capacitiesJson) : DEFAULT_BASELINE.capacities;

        // Set Non-Recyclable to 92% (or Organic if already critical)
        if (capacities.nonRecyclable < 90) {
            capacities.nonRecyclable = 92;
        } else {
            capacities.organic = 94;
        }

        localStorage.setItem('binCapacities', JSON.stringify(capacities));

        renderBinCapacityCards();
        checkAndDisplayCapacityAlerts();
        showToast("⚠️ 90% Capacity Alert Triggered for Waste Bin");
    } catch (e) {
        console.error("Failed to simulate alert:", e);
    }
}

/**
 * Reset Demo to default baseline values
 */
function resetDemoDashboard() {
    try {
        localStorage.setItem('binCapacities', JSON.stringify(DEFAULT_BASELINE.capacities));
        localStorage.setItem('baseCounts', JSON.stringify(DEFAULT_BASELINE.baseItemCounts));
        localStorage.setItem('classificationHistory', JSON.stringify(DEFAULT_BASELINE.sampleHistory));

        initializeDashboard();
        showToast("🔄 Dashboard and demo data reset to default");
    } catch (e) {
        console.error("Failed to reset demo:", e);
    }
}

/**
 * Clear Classification History
 */
function clearClassificationHistory() {
    if (confirm("Are you sure you want to clear all classification history?")) {
        try {
            localStorage.setItem('classificationHistory', JSON.stringify([]));
            renderHistoryTable();
            showToast("🗑️ Classification history cleared");
        } catch (e) {
            console.error("Failed to clear history:", e);
        }
    }
}

/**
 * Helper to show toast message
 */
function showToast(msg) {
    let toast = document.getElementById('dashboardToast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'dashboardToast';
        toast.className = 'toast-notification';
        document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.classList.add('active');
    setTimeout(() => {
        toast.classList.remove('active');
    }, 2800);
}

/**
 * Safe HTML escaping
 */
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
