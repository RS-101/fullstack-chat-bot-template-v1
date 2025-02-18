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

# Updated system prompt with additional intents (removed set_name)
SYSTEM_PROMPT = """
You are an intent classifier. Given a user input, classify it into one of the following intents:
- hello
- help
- goodbye
- info
- merch_tshirt
- merch_cap
- merch_shirt
- merch_watch
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

async def extract_name(user_input: str, use_gemini: bool) -> str:
    """
    Uses the language model to extract the user's first name from their input.
    The prompt instructs the model to respond with only the name.
    """
    NAME_EXTRACTION_PROMPT = "Extract the person's first name from the following text. Respond with only the name."
    
    if use_gemini:
        payload = {
            "contents": [{
                "parts": [{"text": f"{NAME_EXTRACTION_PROMPT}\nText: {user_input}"}]
            }]
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(GEMINI_API_URL, json=payload)
            response_data = response.json()
            try:
                name = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
            except (KeyError, IndexError, TypeError):
                name = ""
        return name
    else:
        response = ollama.chat(
            model="gemma2:2b",
            messages=[
                {"role": "system", "content": NAME_EXTRACTION_PROMPT},
                {"role": "user", "content": user_input},
            ]
        )

        print(response)
        return response['message']['content'].strip()

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
        self.user_data = {}      # Stores variables such as the user's name
        self.chat_history = []   # Tracks conversation history
        self.awaiting_name = False  # Flag to indicate the bot is waiting for the user's name

    def add_node(self, node):
        self.nodes[node.name] = node

    def set_start_node(self, node_name):
        self.current_node = self.nodes.get(node_name)

    async def handle_input(self, user_input, use_gemini):
        # Append the user message to chat history
        self.chat_history.append("User: " + user_input)

        print(f"Current note is :{self.current_node.name}")

        # If we are waiting for the user's name, try to extract it
        if self.awaiting_name:
            extracted_name = await extract_name(user_input, use_gemini)
            if not extracted_name or extracted_name.lower() in ["", "unknown"]:
                response = "I didn't catch your name. Could you please tell me your name?"
                self.chat_history.append("Bot: " + response)
                return response
            else:
                self.user_data["name"] = extracted_name
                self.awaiting_name = False
                response = f"Nice to meet you, {extracted_name}!"
                self.chat_history.append("Bot: " + response)
                # After setting the name, set the current node to greeting
                self.current_node = self.nodes.get("greeting")
                return response

        # Determine the intent using the classifier
        intent = await classify_intent(user_input, use_gemini)
        print(f"Detected Intent: {intent}")

        # If greeted and the name isn't set, ask for the name naturally.
        if intent == "hello" and "name" not in self.user_data:
            self.awaiting_name = True
            response = "Hello! I don't think we've met yet. What's your name?"
            self.chat_history.append("Bot: " + response)
            return response

        response = self.current_node.get_response(intent)

        # Transition between nodes based on the detected intent
        if intent == "help":
            print("Transitioning to help node...")
            self.current_node = self.nodes.get("help")
        elif intent == "goodbye":
            print("Transitioning to goodbye node...")
            self.current_node = self.nodes.get("goodbye")
        elif intent.startswith("merch_"):
            print("Transitioning to merch node...")
            self.current_node = self.nodes.get("merch")
        elif intent == "hello":
            print("Transitioning to greeting node...")
            self.current_node = self.nodes.get("greeting")
        elif intent == "info":
            print("Transitioning to info node...")
            self.current_node = self.nodes.get("info")
        elif intent == "fallback":
            print("Transitioning to info greeting note...")
            self.current_node = self.nodes.get("greeting")


        print(f"New note is :{self.current_node.name}")

        if response:
            # Fill in variables (like {name}) if present
            response = response.format(name=self.user_data.get("name", "there"))
            self.chat_history.append("Bot: " + response)
            return response

        print(f"No predefined response for intent '{intent}'. Querying AI with chat history for context...")
        # Fallback: query the AI directly with chat history context
        if use_gemini:
            history_text = "\n".join(self.chat_history)
            print(f"Chat history: {history_text}")
            combined_prompt = f"Conversation history:\n{history_text}\nUser: {user_input}\n"
            payload = {"contents": [{"parts": [{"text": combined_prompt}]}]}
            async with httpx.AsyncClient() as client:
                ai_response = await client.post(GEMINI_API_URL, json=payload)
                response_data = ai_response.json()
                try:
                    fallback_response = response_data['candidates'][0]['content']['parts'][0]['text']
                except (KeyError, IndexError, TypeError):
                    fallback_response = "I'm sorry, I couldn't process that request."
            self.chat_history.append("Bot: " + fallback_response)
            return fallback_response
        else:
            # For Ollama, build a messages list including the chat history
            messages = [{"role": "system", "content": "Please use the conversation history to provide context."}]
            for entry in self.chat_history:
                if entry.startswith("User:"):
                    messages.append({"role": "user", "content": entry})
                else:
                    messages.append({"role": "assistant", "content": entry})
            messages.append({"role": "user", "content": user_input})
            print(f"Chat history: {messages}")
            ai_response = ollama.chat(model='gemma2:2b', messages=messages)
            fallback_response = ai_response['message']['content']
            self.chat_history.append("Bot: " + fallback_response)
            return fallback_response

# Initialize chatbot and its nodes
bot = ChatBot()

# Greeting Node
greeting_node = ChatNode("greeting")
greeting_node.add_intent("hello", "Hi {name}, how can I help you today?")
greeting_node.add_intent("help", "Sure, I can help! What do you need assistance with?")
greeting_node.add_intent("goodbye", "Goodbye! Have a great day.")
greeting_node.add_intent("info", "I can provide information on various topics. Ask me anything!")

# Help Node
help_node = ChatNode("help")
help_node.add_intent("info", "I can provide detailed help on our services. What would you like to know?")
help_node.add_intent("back", "Returning to the main menu...")
help_node.add_intent("goodbye", "Goodbye! Have a great day.")

# Goodbye Node
goodbye_node = ChatNode("goodbye")
goodbye_node.add_intent("goodbye", "Farewell! See you next time.")

# Merch Node for handling merchandise inquiries
merch_node = ChatNode("merch")
merch_node.add_intent("merch_tshirt", "We have a variety of T-shirts available. Would you like to know more details?")
merch_node.add_intent("merch_cap", "Our caps are stylish and comfortable. Can I provide more information?")
merch_node.add_intent("merch_shirt", "We offer premium shirts in various sizes. Interested in a specific style?")
merch_node.add_intent("merch_watch", "Our watches are elegant and reliable. Are you looking for a specific model?")

# Info Node
info_node = ChatNode("info")
info_node.add_intent("info", "Here's some information: We are a demo chatbot built with node-based logic. How else can I assist you?")

# Add nodes to the chatbot
bot.add_node(greeting_node)
bot.add_node(help_node)
bot.add_node(goodbye_node)
bot.add_node(merch_node)
bot.add_node(info_node)

# Set the starting node
bot.set_start_node("greeting")

@app.post("/api/data")
async def get_data(request: MessageRequest):
    user_input = request.message.strip()
    response = await bot.handle_input(user_input, request.use_gemini)
    print(f"Response: {response}")
    return {"message": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
