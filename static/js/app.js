/**
 * Laptop Price Predictor - Frontend JavaScript
 * Created by: Jahanzaib
 * 
 * Features:
 * - Tab navigation
 * - Form handling with validation
 * - Image upload with drag & drop
 * - API integration for predictions
 * - Chat interface for AI assistance
 */

// ===== Global State =====
const state = {
    currentTab: 'manual',
    selectedImage: null,
    isLoading: false
};

// ===== DOM Elements =====
const elements = {
    tabs: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),
    predictionForm: document.getElementById('prediction-form'),
    resultCard: document.getElementById('result-card'),
    priceValue: document.getElementById('price-value'),
    insightsList: document.getElementById('insights-list'),
    uploadArea: document.getElementById('upload-area'),
    imageInput: document.getElementById('image-input'),
    imagePreview: document.getElementById('image-preview'),
    previewImg: document.getElementById('preview-img'),
    removeImageBtn: document.getElementById('remove-image'),
    aiPredictBtn: document.getElementById('ai-predict-btn'),
    aiResultCard: document.getElementById('ai-result-card'),
    aiResultContent: document.getElementById('ai-result-content'),
    chatForm: document.getElementById('chat-form'),
    chatInput: document.getElementById('chat-input'),
    chatMessages: document.getElementById('chat-messages'),
    helpContent: document.getElementById('help-content'),
    specsGrid: document.getElementById('specs-grid')
};

// ===== Initialize =====
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initForm();
    initImageUpload();
    initChat();
    loadHelpContent();
    loadFormOptions();
});

// ===== Tab Navigation =====
function initTabs() {
    elements.tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.dataset.tab;
            switchTab(tabId);
        });
    });
}

function switchTab(tabId) {
    // Update tab buttons
    elements.tabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabId);
    });
    
    // Update tab contents
    elements.tabContents.forEach(content => {
        content.classList.toggle('active', content.id === `${tabId}-tab`);
    });
    
    state.currentTab = tabId;
}

// ===== Form Handling =====
async function loadFormOptions() {
    try {
        const response = await fetch('/api/options');
        const options = await response.json();
        
        // Populate dropdowns
        populateSelect('company', options.Company);
        populateSelect('typename', options.TypeName);
        populateSelect('cpu', options.Cpu_brand);
        populateSelect('gpu', options.Gpu_brand);
        populateSelect('os', options.os);
        populateSelect('ram', options.Ram);
        populateSelect('ssd', options.SSD, true);
        populateSelect('hdd', options.HDD, true);
        
    } catch (error) {
        console.error('Error loading options:', error);
    }
}

function populateSelect(elementId, options, includeZero = false) {
    const select = document.getElementById(elementId);
    if (!select) return;
    
    // Keep first option (placeholder)
    const placeholder = select.options[0];
    select.innerHTML = '';
    select.appendChild(placeholder);
    
    options.forEach(option => {
        const opt = document.createElement('option');
        opt.value = option;
        
        if (typeof option === 'number') {
            if (option === 0) {
                opt.textContent = elementId === 'ssd' ? 'No SSD' : 
                                  elementId === 'hdd' ? 'No HDD' : '0';
            } else {
                opt.textContent = `${option} GB`;
            }
        } else {
            opt.textContent = option;
        }
        
        select.appendChild(opt);
    });
}

function initForm() {
    elements.predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (state.isLoading) return;
        
        const formData = new FormData(e.target);
        const data = {
            Company: formData.get('Company'),
            TypeName: formData.get('TypeName'),
            Ram: parseInt(formData.get('Ram')) || 8,
            Weight: parseFloat(formData.get('Weight')) || 1.8,
            Touchscreen: formData.get('Touchscreen') ? 1 : 0,
            Ips: formData.get('Ips') ? 1 : 0,
            ppi: parseFloat(formData.get('ppi')) || 141.21,
            Cpu_brand: formData.get('Cpu_brand'),
            HDD: parseInt(formData.get('HDD')) || 0,
            SSD: parseInt(formData.get('SSD')) || 0,
            Gpu_brand: formData.get('Gpu_brand'),
            os: formData.get('os')
        };
        
        await predictPrice(data);
    });
}

async function predictPrice(data) {
    const submitBtn = elements.predictionForm.querySelector('.predict-btn');
    
    try {
        setLoading(true, submitBtn);
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResult(result);
        } else {
            showError(result.error);
        }
        
    } catch (error) {
        showError('Failed to connect to server. Please try again.');
        console.error('Prediction error:', error);
    } finally {
        setLoading(false, submitBtn);
    }
}

function displayResult(result) {
    // Show result card
    elements.resultCard.classList.remove('hidden');
    elements.resultCard.classList.add('success-animation');
    
    // Animate price counter
    animateCounter(elements.priceValue, result.predicted_price);
    
    // Display insights
    elements.insightsList.innerHTML = '';
    result.insights.forEach(insight => {
        const li = document.createElement('li');
        li.textContent = insight;
        elements.insightsList.appendChild(li);
    });
    
    // Scroll to result
    elements.resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function animateCounter(element, target) {
    const duration = 1000;
    const start = 0;
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const current = Math.floor(start + (target - start) * easeOutQuart);
        
        element.textContent = current.toLocaleString();
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

// ===== Image Upload =====
function initImageUpload() {
    // Click to upload
    elements.uploadArea.addEventListener('click', () => {
        elements.imageInput.click();
    });
    
    // File input change
    elements.imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleImageFile(file);
        }
    });
    
    // Drag and drop
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add('dragover');
    });
    
    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.classList.remove('dragover');
    });
    
    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('dragover');
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleImageFile(file);
        }
    });
    
    // Remove image
    elements.removeImageBtn.addEventListener('click', () => {
        removeImage();
    });
    
    // AI Predict button
    elements.aiPredictBtn.addEventListener('click', async () => {
        if (state.selectedImage && !state.isLoading) {
            await predictFromImage();
        }
    });
}

function handleImageFile(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        state.selectedImage = e.target.result;
        elements.previewImg.src = e.target.result;
        elements.uploadArea.classList.add('hidden');
        elements.imagePreview.classList.remove('hidden');
        elements.aiPredictBtn.disabled = false;
    };
    
    reader.readAsDataURL(file);
}

function removeImage() {
    state.selectedImage = null;
    elements.imageInput.value = '';
    elements.previewImg.src = '';
    elements.imagePreview.classList.add('hidden');
    elements.uploadArea.classList.remove('hidden');
    elements.aiPredictBtn.disabled = true;
    elements.aiResultCard.classList.add('hidden');
}

async function predictFromImage() {
    try {
        setLoading(true, elements.aiPredictBtn);
        
        const response = await fetch('/api/predict-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: state.selectedImage
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayAIResult(result.analysis);
        } else {
            showError(result.error);
        }
        
    } catch (error) {
        showError('Failed to analyze image. Please try again.');
        console.error('AI prediction error:', error);
    } finally {
        setLoading(false, elements.aiPredictBtn);
    }
}

function displayAIResult(analysis) {
    elements.aiResultCard.classList.remove('hidden');
    elements.aiResultCard.classList.add('success-animation');
    
    let html = '';
    
    // Brand and Model
    if (analysis.brand || analysis.model) {
        html += `
            <div class="ai-result-item">
                <div class="label">Detected Laptop</div>
                <div class="value">${analysis.brand || 'Unknown'} ${analysis.model || ''}</div>
            </div>
        `;
    }
    
    // Confidence
    if (analysis.confidence) {
        const confidenceColors = {
            high: '#38ef7d',
            medium: '#F2C94C',
            low: '#f56565'
        };
        html += `
            <div class="ai-result-item">
                <div class="label">Confidence Level</div>
                <div class="value" style="color: ${confidenceColors[analysis.confidence]}">${analysis.confidence.toUpperCase()}</div>
            </div>
        `;
    }
    
    // Description
    if (analysis.description) {
        html += `
            <div class="ai-result-item">
                <div class="label">AI Description</div>
                <div class="value" style="font-size: 0.95rem; font-weight: 400;">${analysis.description}</div>
            </div>
        `;
    }
    
    // Predicted Price
    if (analysis.price_formatted) {
        html += `
            <div class="ai-price-display">
                <div class="label" style="margin-bottom: 0.5rem;">Estimated Price</div>
                <div class="price">${analysis.price_formatted}</div>
            </div>
        `;
    }
    
    // Specs Grid
    if (analysis.estimated_specs) {
        const specs = analysis.estimated_specs;
        html += `
            <div style="margin-top: 1rem;">
                <div class="label" style="margin-bottom: 0.75rem; font-size: 0.85rem;">DETECTED SPECIFICATIONS</div>
                <div class="specs-grid-result">
        `;
        
        const specLabels = {
            Company: '🏢 Brand',
            TypeName: '💻 Type',
            Ram: '🧠 RAM',
            Cpu_brand: '⚡ Processor',
            Gpu_brand: '🎮 GPU',
            SSD: '💾 SSD',
            HDD: '📀 HDD',
            os: '🖥️ OS',
            Touchscreen: '👆 Touch',
            Ips: '📺 IPS'
        };
        
        for (const [key, value] of Object.entries(specs)) {
            if (specLabels[key]) {
                let displayValue = value;
                if (key === 'Ram' || key === 'SSD' || key === 'HDD') {
                    displayValue = value === 0 ? 'None' : `${value} GB`;
                } else if (key === 'Touchscreen' || key === 'Ips') {
                    displayValue = value === 1 ? 'Yes' : 'No';
                }
                
                html += `
                    <div class="ai-result-item">
                        <div class="label">${specLabels[key]}</div>
                        <div class="value">${displayValue}</div>
                    </div>
                `;
            }
        }
        
        html += '</div></div>';
    }
    
    elements.aiResultContent.innerHTML = html;
    elements.aiResultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ===== Chat Interface =====
function initChat() {
    elements.chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = elements.chatInput.value.trim();
        if (!message || state.isLoading) return;
        
        // Add user message
        addChatMessage(message, 'user');
        elements.chatInput.value = '';
        
        // Get AI response
        await getAIResponse(message);
    });
}

function addChatMessage(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}-message`;
    
    const avatar = type === 'user' ? '👤' : '🤖';
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${formatMessage(content)}</div>
    `;
    
    elements.chatMessages.appendChild(messageDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function formatMessage(content) {
    // Convert markdown-like formatting to HTML
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>')
        .replace(/- (.*?)(?=<br>|$)/g, '<li>$1</li>')
        .replace(/(<li>.*<\/li>)+/g, '<ul>$&</ul>');
}

async function getAIResponse(query) {
    // Add loading message
    const loadingId = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'chat-message bot-message';
    loadingDiv.id = loadingId;
    loadingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="loading-spinner" style="padding: 0.5rem;">
                <div class="spinner" style="width: 24px; height: 24px;"></div>
            </div>
        </div>
    `;
    elements.chatMessages.appendChild(loadingDiv);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    
    try {
        const response = await fetch('/api/gemini-info', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });
        
        const result = await response.json();
        
        // Remove loading message
        document.getElementById(loadingId)?.remove();
        
        if (result.success) {
            addChatMessage(result.response, 'bot');
        } else {
            addChatMessage(`Sorry, I encountered an error: ${result.error}`, 'bot');
        }
        
    } catch (error) {
        document.getElementById(loadingId)?.remove();
        addChatMessage('Sorry, I could not connect to the server. Please try again.', 'bot');
        console.error('Chat error:', error);
    }
}

// ===== Help Content =====
async function loadHelpContent() {
    try {
        const response = await fetch('/api/help');
        const helpData = await response.json();
        
        // Populate help methods
        let methodsHtml = '';
        helpData.methods.forEach(method => {
            methodsHtml += `
                <div class="help-method">
                    <div class="help-method-icon">${method.icon}</div>
                    <h4>${method.title}</h4>
                    <p>${method.description}</p>
                </div>
            `;
        });
        elements.helpContent.innerHTML = methodsHtml;
        
        // Populate specs explanation
        let specsHtml = '';
        for (const [spec, description] of Object.entries(helpData.specs_explanation)) {
            specsHtml += `
                <div class="spec-item">
                    <h5>${spec}</h5>
                    <p>${description}</p>
                </div>
            `;
        }
        elements.specsGrid.innerHTML = specsHtml;
        
    } catch (error) {
        elements.helpContent.innerHTML = `
            <div class="error-message">
                Failed to load help content. Please refresh the page.
            </div>
        `;
        console.error('Help content error:', error);
    }
}

// ===== Utility Functions =====
function setLoading(loading, button) {
    state.isLoading = loading;
    
    if (button) {
        button.classList.toggle('loading', loading);
        button.disabled = loading;
    }
}

function showError(message) {
    // Create error toast
    const toast = document.createElement('div');
    toast.className = 'error-message';
    toast.style.cssText = `
        position: fixed;
        top: 100px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        max-width: 400px;
        animation: fadeInUp 0.3s ease-out;
    `;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'fadeIn 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ===== Keyboard Shortcuts =====
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + 1-4 for tab switching
    if ((e.ctrlKey || e.metaKey) && e.key >= '1' && e.key <= '4') {
        e.preventDefault();
        const tabs = ['manual', 'ai', 'chat', 'help'];
        const index = parseInt(e.key) - 1;
        if (tabs[index]) {
            switchTab(tabs[index]);
        }
    }
});

// ===== Console Welcome =====
console.log(`
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   🎓 LAPTOP PRICE PREDICTOR                              ║
║   Created by: Jahanzaib                                  ║
║   Data Science Project 2024                              ║
║                                                          ║
║   Features:                                              ║
║   • Manual specification-based prediction                ║
║   • AI image analysis with Gemini                        ║
║   • Interactive chat assistant                           ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
`);
