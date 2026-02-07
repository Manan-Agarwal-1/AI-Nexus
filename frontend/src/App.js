// Main application JavaScript
const API_BASE_URL = 'http://localhost:5000/api';

async function analyzeMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) {
        alert('Please enter a message to analyze');
        return;
    }
    
    // Show loading state
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message })
        });
        
        if (!response.ok) {
            throw new Error('Analysis failed');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing message. Make sure the API server is running.');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze Message';
    }
}

function displayResults(data) {
    const resultsSection = document.getElementById('results');
    resultsSection.style.display = 'block';
    
    // Display alert if present
    const alertDiv = document.getElementById('alert');
    if (data.alert) {
        alertDiv.className = `alert alert-${data.alert.priority}`;
        alertDiv.innerHTML = `
            <h3>${data.alert.title}</h3>
            <p>${data.alert.description}</p>
        `;
        alertDiv.style.display = 'block';
    } else {
        alertDiv.style.display = 'none';
    }
    
    // Display risk score
    const riskScore = Math.round(data.risk_score * 100);
    document.getElementById('riskScore').textContent = `${riskScore}%`;
    document.getElementById('riskLevel').textContent = data.risk_level.toUpperCase();
    document.getElementById('riskLevel').className = `risk-level risk-${data.risk_level}`;
    
    // Display prediction
    const predictionDiv = document.getElementById('prediction');
    predictionDiv.textContent = data.prediction.toUpperCase();
    predictionDiv.className = `prediction prediction-${data.prediction}`;
    
    const confidence = Math.round(data.confidence * 100);
    document.getElementById('confidence').textContent = `Confidence: ${confidence}%`;
    
    // Display risk factors
    const riskFactorsDiv = document.getElementById('riskFactors');
    if (data.risk_factors && data.risk_factors.length > 0) {
        riskFactorsDiv.innerHTML = `
            <h3>⚠️ Risk Factors Detected</h3>
            <ul>
                ${data.risk_factors.map(factor => `<li>${factor}</li>`).join('')}
            </ul>
        `;
    } else {
        riskFactorsDiv.innerHTML = '<p>No specific risk factors detected.</p>';
    }
    
    // Display recommendations
    const recommendationsDiv = document.getElementById('recommendations');
    if (data.alert && data.alert.recommended_actions) {
        recommendationsDiv.innerHTML = `
            <h3>✅ Recommended Actions</h3>
            <ul>
                ${data.alert.recommended_actions.map(action => `<li>${action}</li>`).join('')}
            </ul>
        `;
    } else {
        recommendationsDiv.innerHTML = '';
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Allow Enter key to submit
document.addEventListener('DOMContentLoaded', function() {
    const messageInput = document.getElementById('messageInput');
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            analyzeMessage();
        }
    });
});