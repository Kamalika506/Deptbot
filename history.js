async function loadHistory() {
    try {
        const response = await fetch('/chat_history');
        if (!response.ok) {
            throw new Error('Failed to fetch chat history');
        }
        const data = await response.json();

        const historyList = document.getElementById('history-list');
        historyList.innerHTML = ''; // Clear previous content

        if (data.chat_history && data.chat_history.length > 0) {
            data.chat_history.forEach(entry => {
                const entryElement = document.createElement('li');
                entryElement.innerHTML = `
                    <strong>User:</strong> ${entry.question} <br>
                    <strong>Bot:</strong> ${entry.answer} <br>
                    <small>${entry.timestamp}</small>
                    <button class="delete-btn" onclick="deleteEntry('${entry.timestamp}')">🗑️</button>
                `;
                historyList.appendChild(entryElement);
            });
        } else {
            historyList.innerHTML = '<li>No chat history available.</li>';
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
        const historyList = document.getElementById('history-list');
        historyList.innerHTML = '<li>Error loading chat history. Please try again later.</li>';
    }
}

async function filterByDate() {
    const dateInput = document.getElementById('dateInput').value;
    if (!dateInput) {
        alert('Please select a date to filter.');
        return;
    }

    try {
        const response = await fetch(`/chat_history?date=${dateInput}`);
        if (!response.ok) {
            throw new Error('Failed to fetch filtered chat history');
        }
        const data = await response.json();
        const historyList = document.getElementById('history-list');
        historyList.innerHTML = ''; // Clear previous content

        if (data.chat_history && data.chat_history.length > 0) {
            data.chat_history.forEach(entry => {
                const entryElement = document.createElement('li');
                entryElement.innerHTML = `
                    <strong>User:</strong> ${entry.question} <br>
                    <strong>Bot:</strong> ${entry.answer} <br>
                    <small>${entry.timestamp}</small>
                    <button class="delete-btn" onclick="deleteEntry('${entry.timestamp}')">🗑️</button>
                `;
                historyList.appendChild(entryElement);
            });
        } else {
            historyList.innerHTML = '<li>No chat history available for the selected date.</li>';
        }
    } catch (error) {
        console.error('Error filtering chat history:', error);
        const historyList = document.getElementById('history-list');
        historyList.innerHTML = '<li>Error filtering chat history. Please try again later.</li>';
    }
}

async function deleteEntry(timestamp) {
    if (!confirm('Are you sure you want to delete this chat entry?')) {
        return;
    }

    try {
        const response = await fetch(`/delete_chat_entry?timestamp=${encodeURIComponent(timestamp)}`, {
            method: 'DELETE',
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Failed to delete chat entry');
        }
        
        // Show success message
        alert('Chat entry deleted successfully');
        loadHistory(); // Reload history
    } catch (error) {
        console.error('Error deleting chat entry:', error);
        alert(error.message || 'Failed to delete chat entry. Please try again.');
    }
}

// Load history when the page loads
loadHistory();