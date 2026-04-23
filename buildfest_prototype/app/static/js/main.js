// Main JavaScript file for the Fairness Monitoring Dashboard
// Most functionality is handled inline in templates, but common utilities can go here

console.log('Fairness Monitoring Dashboard loaded');

// Utility function to format numbers
function formatNumber(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

// Utility function to show loading state
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '<p>Loading...</p>';
        element.style.display = 'block';
    }
}

// Utility function to show error
function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="result-error">
                <p><strong>Error:</strong> ${message}</p>
            </div>
        `;
        element.style.display = 'block';
    }
}

