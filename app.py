from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for
from flask_cors import CORS
import os
import json
import pandas as pd
import heapq
import spacy
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from time import sleep
from functools import lru_cache
import hashlib
from dotenv import load_dotenv


app = Flask(__name__)
CORS(app)
load_dotenv()
app.secret_key = os.getenv("SECRET_KEY")

def is_non_it_department_question(user_query):
    # Keywords indicating other departments
    non_it_departments = [
        "cse", "computer science", "computer science engineering",
        "ece", "electronics and communication",
        "eee", "electrical and electronics",
        "mechanical", "civil", "chemical", "bme", "bio medical"
    ]
    
    query_lower = user_query.lower()
    return any(dept in query_lower for dept in non_it_departments)

def contains_gender_pronoun(user_input):
    gender_pronouns = ["he", "she", "his", "her", "him"]
    tokens = user_input.lower().split()
    return any(pronoun in tokens for pronoun in gender_pronouns)


# Hardcoded admin credentials (store in environment variables in production)
ADMIN_EMAIL = "admin@ssn.edu.in"
ADMIN_PASSWORD = "admin123"

class Config:
    SEMANTIC_THRESHOLD = 0.6
    #SEMANTIC_WEIGHT = 0.7  # Weight for semantic similarity in hybrid search
    FEEDBACK_FILE = 'feedback.json'

# ====================== Gemini Configuration ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')

# ====================== NLP Initialization ======================
try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ====================== Data Structures ======================
class BPlusTreeNode:
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.keys = []
        self.children = []

class BPlusTree:
    def __init__(self, order=50):
        self.root = BPlusTreeNode(is_leaf=True)
        self.order = order

    def insert(self, key, value):
        if len(self.root.keys) == (2 * self.order - 1):
            new_root = BPlusTreeNode(is_leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key, value)

    def _insert_non_full(self, node, key, value):
        i = len(node.keys) - 1
        if node.is_leaf:
            node.keys.append((key, value))
            node.keys.sort(key=lambda x: x[0])  # Sort by timestamp
        else:
            while i >= 0 and key < node.keys[i][0]:
                i -= 1
            i += 1
            if len(node.children[i].keys) == (2 * self.order - 1):
                self._split_child(node, i)
                if key > node.keys[i][0]:
                    i += 1
            self._insert_non_full(node.children[i], key, value)

    def _split_child(self, parent, index):
        child = parent.children[index]
        new_node = BPlusTreeNode(is_leaf=child.is_leaf)
        parent.keys.insert(index, child.keys[self.order - 1])
        parent.children.insert(index + 1, new_node)
        new_node.keys = child.keys[self.order:(2 * self.order - 1)]
        child.keys = child.keys[0:(self.order - 1)]
        if not child.is_leaf:
            new_node.children = child.children[self.order:(2 * self.order)]
            child.children = child.children[0:self.order]

    def search_range(self, start_date, end_date):
        """Search for all chats within a date range."""
        results = []
        self._search_range(self.root, start_date, end_date, results)
        return results

    def _search_range(self, node, start_date, end_date, results):
        for key, value in node.keys:
            if start_date <= key <= end_date:
                results.append(value)
        if not node.is_leaf:
            for child in node.children:
                self._search_range(child, start_date, end_date, results)

class HashTable:
    def __init__(self, cache_size=100):
        self.size = 1000
        self.table = [[] for _ in range(self.size)]
        self.cache = {}
        self.cache_size = cache_size

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self._hash(key)
        for kvp in self.table[index]:
            if kvp[0] == key:
                kvp[1] = value
                return
        self.table[index].append([key, value])

    def get(self, key):
        if key in self.cache:
            return self.cache[key]

        index = self._hash(key)
        for kvp in self.table[index]:
            if kvp[0] == key:
                if len(self.cache) >= self.cache_size:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[key] = kvp[1]
                return kvp[1]
        return None

class MaxHeap:
    def __init__(self):
        self.heap = []

    def push(self, score, value):
        heapq.heappush(self.heap, (-score, value))  # Store as tuple with negated score

    def pop(self):
        neg_score, value = heapq.heappop(self.heap)
        return -neg_score, value  # Return actual score and value


# ====================== Data Loading ======================
def load_users():
    if not os.path.exists('users.json'):
        with open('users.json', 'w') as file:
            json.dump([], file)
    with open('users.json', 'r') as file:
        return json.load(file)

def save_users(users):
    with open('users.json', 'w') as file:
        json.dump(users, file, indent=4)

def load_chat_history():
    if not os.path.exists("chats.json"):
        return {}
    with open("chats.json", 'r') as file:
        return json.load(file)

def save_chat_history(chats):
    with open("chats.json", 'w') as file:
        json.dump(chats, file, indent=4)

def load_feedback():
    if not os.path.exists(Config.FEEDBACK_FILE):
        with open(Config.FEEDBACK_FILE, 'w') as file:
            json.dump([], file)
        return []
    
    try:
        with open(Config.FEEDBACK_FILE, 'r') as file:
            # Check if file is empty
            content = file.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupted, reset it
        with open(Config.FEEDBACK_FILE, 'w') as file:
            json.dump([], file)
        return []

def save_feedback(feedback_data):
    with open(Config.FEEDBACK_FILE, 'w') as file:
        json.dump(feedback_data, file, indent=4)

# ====================== Hybrid Search Implementation ======================
csv_path = "dept_ds_new.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, on_bad_lines='skip').dropna(subset=["Question", "Answer"])
    df["Question"] = df["Question"].astype(str).str.lower().str.strip()
    df["Answer"] = df["Answer"].astype(str).str.strip()
else:
    print("⚠️ Warning: CSV file not found! The chatbot may not work correctly.")
    df = pd.DataFrame(columns=["Question", "Answer"])


def semantic_search(query, threshold=Config.SEMANTIC_THRESHOLD, top_k=1):
    """Pure semantic search using sentence embeddings"""
    try:
        # Encode the query
        query_embedding = semantic_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, question_embeddings)[0]
        
        # Find top matches using max heap
        max_heap = MaxHeap()
        for idx, score in enumerate(similarities):
            if score > threshold:
                max_heap.push(score, idx)

        # Prepare results
        results = []
        for _ in range(min(top_k, len(max_heap.heap))):
            score, idx = max_heap.pop()
            results.append({
                "answer": df.iloc[idx]["Answer"],
                "score": float(score),
                "source": "semantic_search"
            })

        return results if results else None
        
    except Exception as e:
        print(f"Semantic search error: {e}")
        return None

def preprocess_query(query):
    """Basic query cleaning"""
    query = query.lower().strip()
    # Remove special characters if needed
    return ' '.join([token.text for token in nlp(query) if not token.is_stop])

# ====================== Gemini Functions ======================
@lru_cache(maxsize=1000)
def get_cache_key(question: str, context: str = None) -> str:
    """Generate a unique cache key for each question-context pair"""
    key = hashlib.md5(question.encode()).hexdigest()
    if context:
        key += hashlib.md5(context.encode()).hexdigest()
    return key

def ask_gemini(question: str, context: str = None) -> str:
    """
    Enhanced Gemini query with:
    - Rate limiting (1 call/second)
    - Response caching
    - Prompt engineering
    """
    cache_key = get_cache_key(question, context)
    
    if hasattr(ask_gemini, 'cache') and cache_key in ask_gemini.cache:
        return ask_gemini.cache[cache_key]
    
    try:
        sleep(1)  # Rate limiting
        
        prompt = f"""You are an expert assistant for SSN College. Answer questions accurately using the context when provided.

        Context (if relevant):
        {context or 'No specific context provided'}

        Question: {question}

        Guidelines:
        - Be concise but thorough
        - If unsure, say "I don't have that information"
        - For lists, use bullet points
        - For comparisons, use tables
        - Maintain a professional tone
        Answer:"""
        
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 500
            }
        )
        
        answer = response.text
        
        if not hasattr(ask_gemini, 'cache'):
            ask_gemini.cache = {}
        
        ask_gemini.cache[cache_key] = answer
        return answer
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

# ====================== Chatbot Core Functions ======================
def decompose_query(query):
    """Break compound questions into sub-questions using NLP"""
    doc = nlp(query.lower())
    sub_questions = []
    buffer = []
    
    for token in doc:
        if token.text in {"and", "also", "too"} and buffer:
            sub_questions.append(" ".join(buffer))
            buffer = []
        else:
            buffer.append(token.text)
    
    if buffer:
        sub_questions.append(" ".join(buffer))
    
    return sub_questions if len(sub_questions) > 1 else [query]

def expand_query(query):
    """Add synonyms and related terms to the query"""
    synonyms = {
        "fee": ["cost", "tuition", "payment"],
        "course": ["subject", "class", "module"]
    }
    for term, syns in synonyms.items():
        if term in query:
            query += " " + " ".join(syns)
    return query


def get_related_questions(query, top_n=3, similarity_threshold=0.4):
    """Get semantically related questions from the dataset"""
    query_embedding = semantic_model.encode([query])
    similarities = cosine_similarity(query_embedding, question_embeddings)[0]
    
    related_questions = []
    for idx in similarities.argsort()[::-1]:
        if similarities[idx] > similarity_threshold and df.iloc[idx]['Question'].lower() != query.lower():
            related_questions.append({
                "question": df.iloc[idx]['Question'],
                "similarity": float(similarities[idx])
            })
            if len(related_questions) >= top_n:
                break
    return related_questions

def get_semantic_context(query: str) -> str:
    """Get relevant context from similar questions"""
    query_embedding = semantic_model.encode([query])
    similarities = cosine_similarity(query_embedding, question_embeddings)[0]
    
    context_parts = []
    for idx in similarities.argsort()[-3:][::-1]:
        if similarities[idx] > 0.3:
            context_parts.append(
                f"Related Question: {df.iloc[idx]['Question']}\n"
                f"Related Answer: {df.iloc[idx]['Answer']}"
            )
    
    return "\n\n".join(context_parts) if context_parts else None

def update_chat_history(question: str, answer: str):
    """Update all chat history systems"""
    try:
        # Get current user email from session
        user_email = session.get("email")
        if not user_email:
            print("Warning: No email in session when updating chat history")
            return

        # Load existing history
        chat_history_data = load_chat_history()
        
        # Create new chat entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_entry = {
            "question": question,
            "answer": answer,
            "timestamp": timestamp
        }

        # Initialize user's chat history if it doesn't exist
        if user_email not in chat_history_data:
            chat_history_data[user_email] = []

        # Insert new entry at the beginning of the list
        chat_history_data[user_email].insert(0, chat_entry)  # Change made here
        
        # Save to file
        with open("chats.json", 'w') as file:
            json.dump(chat_history_data, file, indent=4)
        
        # Debug print
        print(f"Chat history updated for {user_email} at {timestamp}")
        
    except Exception as e:
        print(f"Error updating chat history: {str(e)}")
        raise  # Re-raise the exception for debugging

    

# ====================== Data Initialization ======================
semantic_model = SentenceTransformer('all-mpnet-base-v2')
question_embeddings = semantic_model.encode(df["Question"].tolist())

#bplus_tree = BPlusTree(order=50)
hash_table = HashTable()
max_heap = MaxHeap()
chat_bplus_tree = BPlusTree(order=50)

for _, row in df.iterrows():
    #bplus_tree.insert(row["Question"], (row["Question"], row["Answer"]))
    hash_table.insert(row["Question"], row["Answer"])

# ====================== Flask Routes ======================


@app.before_request
def check_admin_access():
    admin_routes = ['/admin_dashboard', '/admin/users', '/admin/feedback', '/admin/feedback_stats']
    if request.path in admin_routes and not session.get('is_admin'):
        return jsonify({"error": "Admin access required"}), 403

@app.route('/admin_login', methods=['POST'])
def admin_login():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    
    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        session["logged_in"] = True
        session["email"] = email
        session["is_admin"] = True
        return jsonify({"success": True})
    
    return jsonify({"success": False, "error": "Invalid admin credentials"})

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get("is_admin"):
        return jsonify({"error": "Admin access required"}), 403
    return render_template('admin_dashboard.html')

@app.route('/admin/users')
def get_all_users():
    if not session.get("is_admin"):
        return jsonify({"error": "Admin access required"}), 403
    
    users = load_users()
    safe_users = [{"email": user["email"], "signup_date": user.get("signup_date", "N/A")} 
                 for user in users if not user.get("is_admin", False)]
    
    return jsonify({"users": safe_users})


@app.route('/admin/feedback_stats')
def feedback_stats():
    if not session.get("is_admin"):
        return jsonify({"error": "Admin access required"}), 403
    
    feedback_data = load_feedback()
    
    if not feedback_data:
        return jsonify({
            "total_feedback": 0,
            "average_rating": 0,
            "rating_counts": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "common_issues": []
        })
    
    # Calculate stats
    total = len(feedback_data)
    average = sum(fb['rating'] for fb in feedback_data) / total
    rating_counts = {i: 0 for i in range(1, 6)}
    for fb in feedback_data:
        rating_counts[fb['rating']] += 1
    
    # Get low-rated feedback comments
    low_rating_comments = [
        fb['comments'] for fb in feedback_data 
        if fb['rating'] <= 2 and fb.get('comments', '').strip()
    ]
    
    return jsonify({
        "total_feedback": total,
        "average_rating": round(average, 1),
        "rating_counts": rating_counts,
        "common_issues": low_rating_comments[:5]  # Top 5 issues
    })

@app.route('/admin/feedback')
def get_all_feedback():
    if not session.get("is_admin"):
        return jsonify({"error": "Admin access required"}), 403
    
    rating_filter = request.args.get('rating', type=str)
    time_filter = request.args.get('time', type=str)
    
    feedback_data = load_feedback()
    
    # Apply filters
    if rating_filter and rating_filter.isdigit():
        rating = int(rating_filter)
        feedback_data = [fb for fb in feedback_data if fb["rating"] == rating]
    
    if time_filter and time_filter != 'all':
        now = datetime.now()
        if time_filter == 'today':
            today = now.date()
            feedback_data = [fb for fb in feedback_data 
                           if datetime.strptime(fb["timestamp"], "%Y-%m-%d %H:%M:%S").date() == today]
        elif time_filter == 'week':
            week_ago = now - timedelta(days=7)
            feedback_data = [fb for fb in feedback_data 
                           if datetime.strptime(fb["timestamp"], "%Y-%m-%d %H:%M:%S") >= week_ago]
        elif time_filter == 'month':
            month_ago = now - timedelta(days=30)
            feedback_data = [fb for fb in feedback_data 
                           if datetime.strptime(fb["timestamp"], "%Y-%m-%d %H:%M:%S") >= month_ago]
    
    return jsonify({
        "feedback": feedback_data
    })


@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    users = load_users()
    if any(user["email"] == email for user in users):
        return jsonify({"success": False, "error": "Email already registered"})
    users.append({
        "email": email, 
        "password": password,
        "is_admin": False,
        "signup_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_users(users)
    return jsonify({"success": True})


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    users = load_users()

    # Admin login
    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        session["logged_in"] = True
        session["email"] = email
        session["is_admin"] = True
        return jsonify({"success": True, "is_admin": True})

    # Regular user login
    user = next((user for user in users if user["email"] == email and user["password"] == password), None)
    if user:
        session["logged_in"] = True
        session["email"] = email
        session["is_admin"] = False
        return jsonify({"success": True, "is_admin": False})

    return jsonify({"success": False, "error": "Invalid email or password"})




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chatbot')
def chatbot():
    if not session.get("logged_in"):
        return redirect(url_for("index"))
    if session.get("is_admin"):
        return redirect(url_for("admin_dashboard"))
    return render_template('chatbot.html')


@app.route('/history')
def history():
    if not session.get("logged_in"):
        return redirect(url_for("index"))
    return render_template('history.html')

@app.route('/bin')
def bin():
    if not session.get("logged_in"):
        return redirect(url_for("index"))
    return render_template('bin.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/logout', methods=['POST'])
def logout():
    try:
        session.clear()
        return jsonify({
            "success": True,
            "message": "Logged out successfully",
            "redirect": url_for("index")
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    if "logged_in" not in session:
        return jsonify({"error": "User not logged in"}), 401

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        rating = data.get('rating')
        comments = data.get('comments', '')
        question = data.get('question', '')  # Store the question that was asked
        
        if not rating or not 1 <= int(rating) <= 5:
            return jsonify({"error": "Invalid rating (1-5 required)"}), 400

        feedback_data = load_feedback()
        feedback_data.append({
            "user_email": session.get("email"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rating": int(rating),
            "comments": comments,
            "question": question  # Store the related question
        })
        
        save_feedback(feedback_data)
        return jsonify({"success": True})
        
    except Exception as e:
        print(f"Feedback error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    if "logged_in" not in session or not session.get("logged_in"):
        return jsonify({"error": "User not logged in"}), 401

    user_input = request.json.get("user_input")
    user_email = session.get("email")
    prev_input = session.get("prev_input", "")

    user_query = request.json.get('question', '').strip()
    if not user_query:
        return jsonify({"answer": "Please ask a question."})

    user_query_lower = user_query.lower()

    #  Restrict to IT department questions only
    if is_non_it_department_question(user_query_lower):
        return jsonify({
            "answer": "This bot doesn't answer other department-related questions. Please ask something related to the IT department.",
            "source": "department_filter",
            "recommendations": []
        })

    #  Check for gender pronouns
    if contains_gender_pronoun(user_query_lower):
        if prev_input:
            user_query = f"In reference to the previous question: '{prev_input}', the user now says: '{user_query}'"
        else:
            return jsonify({
                "answer": "Could you please clarify who you're referring to?"
            })

    #  Store current input for future reference
    session["prev_input"] = user_query

    #  Start building the response
    response = {
        "answer": "",
        "source": "",
        "recommendations": []
    }

    # 1️⃣ Try exact match
    exact_answer = hash_table.get(user_query.lower())
    if exact_answer:
        response["answer"] = exact_answer
        response["source"] = "exact_match"
    else:
        # 2️⃣ Try semantic search
        semantic_results = semantic_search(user_query)
        if semantic_results:
            response.update(semantic_results[0])
        else:
            # 3️⃣ Fallback to Gemini
            try:
                context = get_semantic_context(user_query)
                gemini_answer = ask_gemini(user_query, context)
                response["answer"] = gemini_answer or "I couldn't find an answer."
                response["source"] = "gemini"
            except Exception as e:
                print(f"Gemini Error: {e}")
                response["answer"] = "Sorry, I'm having trouble answering right now."
                response["source"] = "error"

    #  Add related question suggestions
    response["recommendations"] = get_related_questions(user_query)

    #  Update chat history
    try:
        update_chat_history(user_query, response["answer"])
    except Exception as e:
        print(f"Failed to update chat history: {str(e)}")

    return jsonify(response)


@app.route('/delete_chat_entry', methods=['DELETE'])
def delete_chat_entry():
    if "logged_in" not in session or not session.get("logged_in"):
        return jsonify({"error": "User not logged in"}), 401

    timestamp = request.args.get('timestamp')
    user_email = session.get("email")
    
    if not timestamp:
        return jsonify({"error": "Timestamp parameter is required"}), 400

    chat_history_data = load_chat_history()
    
    if user_email not in chat_history_data:
        return jsonify({"error": "No chat history found for user"}), 404

    initial_count = len(chat_history_data[user_email])
    chat_history_data[user_email] = [
        entry for entry in chat_history_data[user_email] 
        if entry['timestamp'] != timestamp
    ]
    
    if len(chat_history_data[user_email]) == initial_count:
        return jsonify({"error": "No entry found with that timestamp"}), 404

    save_chat_history(chat_history_data)
    
    return jsonify({"success": True, "message": "Entry deleted successfully"})


@app.route('/chat_history', methods=['GET'])
def chat_history():
    if "logged_in" not in session or not session.get("logged_in"):
        return jsonify({"error": "User not logged in"}), 401

    user_email = session.get("email")
    date_filter = request.args.get("date")

    chat_history_data = load_chat_history()

    if user_email not in chat_history_data:
        return jsonify({"chat_history": []})

    user_chats = chat_history_data[user_email]

    # Filter by date if requested
    if date_filter:
        try:
            date_start = datetime.strptime(date_filter, "%Y-%m-%d").strftime("%Y-%m-%d 00:00:00")
            date_end = datetime.strptime(date_filter, "%Y-%m-%d").strftime("%Y-%m-%d 23:59:59")

            user_chats = [
                chat for chat in user_chats
                if date_start <= chat["timestamp"] <= date_end
            ]
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."})

    #  Sort chats by timestamp in descending order
    user_chats.sort(key=lambda x: x["timestamp"], reverse=True)

    return jsonify({"chat_history": user_chats})


if __name__ == '__main__':
    app.run(debug=True)


