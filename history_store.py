import json

def save_chat_history(history):
    with open("chat_history.json", "w") as file:
        json.dump(history, file)

def load_chat_history():
    try:
        with open("chat_history.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def add_chat_message(message):
    history = load_chat_history()
    history.append(message)
    save_chat_history(history)
