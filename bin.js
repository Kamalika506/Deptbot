document.addEventListener('DOMContentLoaded', function () {
    loadDeletedHistory();
});

function loadDeletedHistory() {
    let deletedHistory = JSON.parse(localStorage.getItem('deletedHistory')) || [];
    let binList = document.getElementById('bin-list');
    binList.innerHTML = '';

    deletedHistory.forEach(msg => {
        let li = document.createElement('li');
        li.textContent = msg;
        binList.appendChild(li);
    });
}
