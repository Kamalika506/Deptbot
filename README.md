# 🤖 Deptbot — Departmental Information & Semantic Search Assistant

**Deptbot** is an interactive departmental information assistant designed for students and faculty. It provides instant automated query resolution, semantic search–based document retrieval, interaction history management, and a dedicated admin interface to collect feedback for continuous accuracy improvements.

---

## ✨ Features

* **💬 Instant Query Resolution:** Interactive chatbot interface (`chatbot.html`) for quick student and faculty inquiries.
* **🔍 Semantic Search Retrieval:** Utilizes vector or contextual retrieval models to pull relevant departmental files and documents.
* **📊 Admin Dashboard:** Dedicated panel (`admin_dashboard.html`) to oversee system activity, review queries, and manage knowledge data.
* **📜 Query & Chat History:** Logged session tracking (`history_store.py`, `history.html`) for context persistence and user review.
* **🗑️ Soft Delete / Trash Bin:** Modular file & history cleanup handling via dedicated bin components (`bin.html`, `bin.js`).

---

## 🛠️ Tech Stack

* **Backend:** Python (Flask/FastAPI API handling via `app.py`, data store management via `history_store.py`)
* **Frontend:** Standard Web Stack (HTML5, CSS3, JavaScript ES6)
* **Styling:** Custom responsive CSS (`chatbot.css`, `styles.css`, `history.css`, `bin.css`)

---

## 📁 Project Structure

```text
Deptbot/
├── app.py                  # Main backend server application & API endpoints
├── history_store.py        # History storage management module
├── index.html              # Main landing page / entry portal
├── admin_dashboard.html    # Administrative control panel
├── chatbot.html            # Main Chatbot interface
├── history.html            # Past conversation logs viewer
├── bin.html                # Recycled/Deleted logs viewer
├── app.js                  # Main client-side script
├── chatbot.js              # Chat interface interactivity & API calls
├── history.js              # Fetching and rendering past session history
├── bin.js                  # Bin handling and recovery/deletion logic
└── *.css                   # Interface stylesheets
