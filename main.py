from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import ollama
import httpx
import os

# Boolean flag to toggle between Gemini API and Ollama
USE_GEMINI = True

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# FastAPI App
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class MessageRequest(BaseModel):
    message: str
    use_gemini: bool = USE_GEMINI

SYSTEM_PROMPT = """
You are an intent classifier. Given a user input, classify it into one of the following intents:
- hello
- help
- goodbye
- info
- fallback (if it doesn't match any intent)

Respond only with the intent name.
"""

async def classify_intent_ollama(user_input: str) -> str:
    """Classifies intent using Ollama."""
    response = ollama.chat(
        model="gemma2:2b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
    )
    return response['message']['content'].strip().lower()

async def classify_intent_gemini(user_input: str) -> str:
    """Classifies intent using Gemini API."""
    payload = {
        "contents": [{
            "parts": [{"text": f"{SYSTEM_PROMPT}\nUser input: {user_input}"}]
        }]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(GEMINI_API_URL, json=payload)
        response_data = response.json()
    
    try:
        intent = response_data['candidates'][0]['content']['parts'][0]['text'].strip().lower()
    except (KeyError, IndexError, TypeError):
        intent = "fallback"
    
    return intent

async def classify_intent(user_input: str, use_gemini: bool) -> str:
    """Determines which intent classification method to use."""
    if use_gemini:
        return await classify_intent_gemini(user_input)
    return await classify_intent_ollama(user_input)

class ChatNode:
    def __init__(self, name):
        self.name = name
        self.intents = {}

    def add_intent(self, intent, response):
        self.intents[intent] = response

    def get_response(self, intent):
        return self.intents.get(intent, None)

class ChatBot:
    def __init__(self):
        self.nodes = {}
        self.current_node = None

    def add_node(self, node):
        self.nodes[node.name] = node

    def set_start_node(self, node_name):
        self.current_node = self.nodes.get(node_name)

    async def handle_input(self, user_input, use_gemini):
        if not self.current_node:
            return "Error: No starting node set."

        intent = await classify_intent(user_input, use_gemini)
        print(f"Intent: {intent}")
        response = self.current_node.get_response(intent)
        if response:
            return response

        print(f"No predefined response for intent '{intent}'. Querying AI...")
        
        if use_gemini:
            payload = {"contents": [{"parts": [{"text": user_input}]}]}
            async with httpx.AsyncClient() as client:
                ai_response = await client.post(GEMINI_API_URL, json=payload)
                response_data = ai_response.json()
                try:
                    return response_data['candidates'][0]['content']['parts'][0]['text']
                except (KeyError, IndexError, TypeError):
                    return "I'm sorry, I couldn't process that request."
        else:
            ai_response = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': user_input}])
            return ai_response['message']['content']

# Initialize chatbot
bot = ChatBot()

greeting_node = ChatNode("greeting")
help_node = ChatNode("help")
goodbye_node = ChatNode("goodbye")

greeting_node.add_intent("hello", "Hi there! How can I assist you?")
greeting_node.add_intent("help", "Sure, I can help! What do you need assistance with?")
greeting_node.add_intent("goodbye", "Goodbye! Have a great day.")

help_node.add_intent("info", "I can provide information on various topics. Ask me anything!")
help_node.add_intent("back", "Returning to greeting...")

goodbye_node.add_intent("bye", "Farewell! See you next time.")

bot.add_node(greeting_node)
bot.add_node(help_node)
bot.add_node(goodbye_node)

bot.set_start_node("greeting")

@app.post("/api/data")
async def get_data(request: MessageRequest):
    print(GEMINI_API_KEY)
    user_input = request.message.strip()
    response = await bot.handle_input(user_input, request.use_gemini)
    print(f"Response: {response}")
    return {"message": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
