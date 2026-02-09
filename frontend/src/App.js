// Main application JavaScript
const API_BASE_URL = 'http://localhost:5000/api';

async function analyzeMessage() {
    console.log('analyzeMessage() called');
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) {
        alert('Please enter a message to analyze');
        return;
    }
    
    console.log('Message to analyze:', message);
    
    // Show loading state
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    
    try {
        console.log('Sending request to:', `${API_BASE_URL}/analyze`);
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message })
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('API Error Response:', errorText);
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('API Response:', data);
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        let errorMsg = 'Error analyzing message. Make sure the API server is running.';
        if (error.message) {
            errorMsg = `Error: ${error.message}`;
        }
        alert(errorMsg);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze Message';
    }
}

function setSample(text) {
    const messageInput = document.getElementById('messageInput');
    messageInput.value = text;
    messageInput.focus();
}

function displayResults(data) {
    console.log('displayResults() called with:', data);
    const resultsSection = document.getElementById('results');
    if (!resultsSection) {
        console.error('results div not found!');
        return;
    }
    resultsSection.style.display = 'block';
    
    // Display alert if present
    const alertDiv = document.getElementById('alert');
    if (data.alert && data.alert.priority) {
        alertDiv.className = `alert alert-${data.alert.priority}`;
        alertDiv.innerHTML = `
            <h3>${data.alert.title || 'Alert'}</h3>
            <p>${data.alert.description || 'Potential scam detected'}</p>
        `;
        alertDiv.style.display = 'block';
    } else {
        alertDiv.style.display = 'none';
    }
    
    // Display risk score
    const riskScore = Math.round(data.risk_score * 100);
    const riskScoreEl = document.getElementById('riskScore');
    const riskLevelEl = document.getElementById('riskLevel');
    if (riskScoreEl) riskScoreEl.textContent = `${riskScore}%`;
    if (riskLevelEl) {
        riskLevelEl.textContent = data.risk_level.toUpperCase();
        riskLevelEl.className = `risk-level risk-${data.risk_level}`;
    }
    console.log(`Risk Score: ${riskScore}%, Level: ${data.risk_level}`);
    
    // Display prediction
    const predictionDiv = document.getElementById('prediction');
    if (predictionDiv) {
        predictionDiv.textContent = data.prediction.toUpperCase();
        predictionDiv.className = `prediction prediction-${data.prediction}`;
    }
    
    const confidence = Math.round(data.confidence * 100);
    const confidenceEl = document.getElementById('confidence');
    if (confidenceEl) confidenceEl.textContent = `Confidence: ${confidence}%`;
    
    // Display risk factors
    const riskFactorsDiv = document.getElementById('riskFactors');
    if (riskFactorsDiv) {
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
    }
    
    // Display recommendations
    const recommendationsDiv = document.getElementById('recommendations');
    if (recommendationsDiv) {
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
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function quickTest() {
    console.log('quickTest() called');
    // Simple direct test without UI
    fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: 'Free money now click here'})
    })
    .then(r => {
        console.log('Response status:', r.status);
        return r.json();
    })
    .then(data => {
        console.log('Response data:', data);
        console.log('Has alert:', !!data.alert);
        console.log('Risk level:', data.risk_level);
        console.log('Risk score:', data.risk_score);
        alert(`SUCCESS!\nRisk: ${data.risk_level}\nScore: ${data.risk_score}\nAlert: ${data.alert ? 'YES' : 'NO'}`);
    })
    .catch(err => {
        console.error('Fetch error:', err);
        alert('ERROR: ' + err.message);
    });
}

// Allow Enter key to submit
document.addEventListener('DOMContentLoaded', function() {
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                analyzeMessage();
            }
        });
    }
});