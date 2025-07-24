let currentRating = 0;
let lastMessageTimestamp = null;

const chatBox = document.getElementById('chat-box');
const userMessageInput = document.getElementById('user-message');
const sendMessageBtn = document.getElementById('send-message');
const logoutBtn = document.getElementById('logout-btn');
const feedbackModal = document.getElementById('feedback-modal');
const ratingStars = document.querySelectorAll('.rating-star');

sendMessageBtn.addEventListener('click', sendMessage);
logoutBtn.addEventListener('click', logout);

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

function sendMessage() {
    const message = userMessageInput.value.trim();
    if (!message) return;

    addMessageToChat(message, 'user');
    userMessageInput.value = '';
    showTypingIndicator();

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: message })
    })
    .then(response => response.json())
    .then(data => {
        removeTypingIndicator();
        if (data.error) {
            addMessageToChat(data.error, 'bot', 'error');
        } else {
            lastMessageTimestamp = new Date().toISOString();
            addMessageToChat(data.answer, 'bot', data.source);
            
            // Display recommendations if they exist
            if (data.recommendations && data.recommendations.length > 0) {
                showRecommendations(data.recommendations);
            } else {
                hideRecommendations();
            }
        }
    })
    .catch(error => {
        removeTypingIndicator();
        addMessageToChat("Sorry, there was an error processing your request.", 'bot', 'error');
        console.error('Error:', error);
    });
}

// Add these new functions
function showRecommendations(recommendations) {
    const container = document.getElementById('recommendations-container');
    container.innerHTML = '<div class="recommendations-title">You might also ask:</div>';
    
    recommendations.forEach(rec => {
        const item = document.createElement('div');
        item.className = 'recommendation-item';
        item.textContent = rec.question;
        item.onclick = () => {
            userMessageInput.value = rec.question;
            sendMessage();
        };
        container.appendChild(item);
    });
    
    container.style.display = 'block';
}

function hideRecommendations() {
    document.getElementById('recommendations-container').style.display = 'none';
}

function addMessageToChat(message, sender, source = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-bubble ${sender}-bubble`;

    const messageContent = document.createElement('p');
    messageContent.textContent = message;
    messageDiv.appendChild(messageContent);

    if (source) {
        const sourceTag = document.createElement('div');
        sourceTag.className = 'source-tag';
        sourceTag.textContent = getSourceLabel(source);
        messageDiv.appendChild(sourceTag);
    }

    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function getSourceLabel(source) {
    const sources = {
        'exact_match': 'Exact match from database',
        'hybrid_search': 'Recommended answer',
        'gemini': 'AI generated response',
        'error': 'System message'
    };
    return sources[source] || source;
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator bot-bubble chat-bubble';
    typingDiv.id = 'typing-indicator';

    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'typing-dot';
        typingDiv.appendChild(dot);
    }

    chatBox.appendChild(typingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

function loadChatHistory() {
    fetch('/history')
        .then(response => response.json())
        .then(data => {
            data.messages.forEach(msg => {
                addMessageToChat(msg.message, msg.sender, msg.source || null);
            });
        })
        .catch(error => {
            console.error('Error loading chat history:', error);
        });
}

function showFeedbackModal() {
    feedbackModal.classList.remove('hidden');
}

function hideFeedbackModal() {
    feedbackModal.classList.add('hidden');
    currentRating = 0;
    updateRatingStars();
    document.getElementById('feedback-comments').value = '';
}

function setRating(rating) {
    currentRating = rating;
    updateRatingStars();
}

function updateRatingStars() {
    ratingStars.forEach(star => {
        const starRating = parseInt(star.getAttribute('data-rating'));
        star.textContent = starRating <= currentRating ? '★' : '☆';
        star.style.color = starRating <= currentRating ? '#e67e22' : '#d1d5db';
    });
}

function submitFeedback() {
    if (!currentRating) {
        alert('Please select a rating');
        return;
    }

    if (!lastMessageTimestamp) {
        alert('No recent message to provide feedback on');
        return;
    }

    const comments = document.getElementById('feedback-comments').value;

    fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            timestamp: lastMessageTimestamp,
            rating: currentRating,
            comments: comments
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Thank you for your feedback!');
            hideFeedbackModal();
        } else {
            alert('Failed to submit feedback');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Failed to submit feedback');
    });
}

function logout() {
    fetch('/logout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            window.location.href = '/';
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}




