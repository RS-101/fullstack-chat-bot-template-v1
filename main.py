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

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System prompt (all lower-case)
SYSTEM_PROMPT = """
You are an intent classifier. Given a user input, classify it into one of the following intents:
- greeting
- i_am_fine
- book_restaurant
- restaurant_opening_hours
- job_opportunities
- cancel
- help
- i_dont_know
- yes
- goodbye
- no
- fallback

Respond only with the intent name.
"""

class MessageRequest(BaseModel):
    message: str
    use_gemini: bool = USE_GEMINI

# ---------------------------
# Intent Classification Functions
# ---------------------------
async def classify_intent_ollama(user_input: str) -> str:
    response = ollama.chat(
        model="gemma2:2b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
    )
    return response['message']['content'].strip()

async def classify_intent_gemini(user_input: str) -> str:
    payload = {
        "contents": [{
            "parts": [{"text": f"{SYSTEM_PROMPT}\nUser input: {user_input}"}]
        }]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(GEMINI_API_URL, json=payload)
        response_data = response.json()
    try:
        intent = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
    except (KeyError, IndexError, TypeError):
        intent = "fallback"
    return intent

async def classify_intent(user_input: str, use_gemini: bool) -> str:
    if use_gemini:
        return await classify_intent_gemini(user_input)
    return await classify_intent_ollama(user_input)

# ---------------------------
# Extraction Helper Functions
# ---------------------------
def clean_extracted_value(value: str) -> str:
    if value and value.lower().startswith("there is no"):
        return ""
    return value

async def extract_date(user_input: str, use_gemini: bool) -> str:
    prompt = "Extract the date from the following text. Respond with only the date."
    if use_gemini:
        payload = {"contents": [{"parts": [{"text": f"{prompt}\nText: {user_input}"}]}]}
        async with httpx.AsyncClient() as client:
            response = await client.post(GEMINI_API_URL, json=payload)
            response_data = response.json()
            try:
                date_val = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
            except (KeyError, IndexError, TypeError):
                date_val = ""
        return clean_extracted_value(date_val)
    else:
        response = ollama.chat(
            model="gemma2:2b",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ]
        )
        return clean_extracted_value(response['message']['content'].strip())

async def extract_time(user_input: str, use_gemini: bool) -> str:
    prompt = "Extract the time from the following text. Respond with only the time."
    if use_gemini:
        payload = {"contents": [{"parts": [{"text": f"{prompt}\nText: {user_input}"}]}]}
        async with httpx.AsyncClient() as client:
            response = await client.post(GEMINI_API_URL, json=payload)
            response_data = response.json()
            try:
                time_val = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
            except (KeyError, IndexError, TypeError):
                time_val = ""
        return clean_extracted_value(time_val)
    else:
        response = ollama.chat(
            model="gemma2:2b",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ]
        )
        return clean_extracted_value(response['message']['content'].strip())

async def extract_number(user_input: str, use_gemini: bool) -> str:
    prompt = "Extract the number of people from the following text. Respond with only the number."
    if use_gemini:
        payload = {"contents": [{"parts": [{"text": f"{prompt}\nText: {user_input}"}]}]}
        async with httpx.AsyncClient() as client:
            response = await client.post(GEMINI_API_URL, json=payload)
            response_data = response.json()
            try:
                number_val = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
            except (KeyError, IndexError, TypeError):
                number_val = ""
        return clean_extracted_value(number_val)
    else:
        response = ollama.chat(
            model="gemma2:2b",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ]
        )
        return clean_extracted_value(response['message']['content'].strip())

# ---------------------------
# Chat Node and ChatBot Classes
# ---------------------------
class ChatNode:
    def __init__(self, name):
        self.name = name
        self.intents = {}
    
    def add_intent(self, intent, response):
        self.intents[intent.lower()] = response

    def get_response(self, intent):
        return self.intents.get(intent.lower(), None)

class ChatBot:
    def __init__(self):
        self.nodes = {}
        self.user_data = {}      # Stores booking details (date, time, number, etc.)
        self.chat_history = []   # Conversation history
        self.pending_slot = None # Which slot (date, time, or number) is currently awaited

    def add_node(self, node: ChatNode):
        self.nodes[node.name] = node

    async def handle_input(self, user_input, use_gemini):
        self.chat_history.append("User: " + user_input)
        print(f"User Input: {user_input}")

        # If waiting for a pending slot (e.g. date, time, or number), process this input first.
        if self.pending_slot:
            if self.pending_slot == "date":
                extracted = await extract_date(user_input, use_gemini)
                if not extracted:
                    response = "I didn't catch the date. When do you want to go?"
                    self.chat_history.append("Bot: " + response)
                    return response
                self.user_data["date"] = extracted
            elif self.pending_slot == "time":
                extracted = await extract_time(user_input, use_gemini)
                if not extracted:
                    response = "I didn't catch the time. What time do you want to go?"
                    self.chat_history.append("Bot: " + response)
                    return response
                self.user_data["time"] = extracted
            elif self.pending_slot == "number":
                extracted = await extract_number(user_input, use_gemini)
                if not extracted:
                    response = "I didn't catch the number of people. How many people will be going?"
                    self.chat_history.append("Bot: " + response)
                    return response
                self.user_data["number"] = extracted

            # Clear the pending slot after successful extraction.
            self.pending_slot = None

            # Check for the next missing slot and ask for it immediately.
            if not self.user_data.get("date"):
                self.pending_slot = "date"
                response = "When do you want to go?"
                self.chat_history.append("Bot: " + response)
                return response
            if not self.user_data.get("time"):
                self.pending_slot = "time"
                response = "What time do you want to go?"
                self.chat_history.append("Bot: " + response)
                return response
            if not self.user_data.get("number"):
                self.pending_slot = "number"
                response = "How many people will be going?"
                self.chat_history.append("Bot: " + response)
                return response

            # If all details are now collected, confirm the booking.
            response = self.nodes["booking"].get_response("book_restaurant")
            response = response.format(date=self.user_data["date"],
                                       time=self.user_data["time"],
                                       number=self.user_data["number"])
            self.chat_history.append("Bot: " + response)
            return response

        # Classify intent if no pending slot.
        intent = (await classify_intent(user_input, use_gemini)).lower()
        print(f"Detected Intent: {intent}")

        if intent == "book_restaurant":
            # Try to extract booking details.
            extracted_date = await extract_date(user_input, use_gemini)
            if extracted_date:
                self.user_data["date"] = extracted_date
            extracted_time = await extract_time(user_input, use_gemini)
            if extracted_time:
                self.user_data["time"] = extracted_time
            extracted_number = await extract_number(user_input, use_gemini)
            if extracted_number:
                self.user_data["number"] = extracted_number

            if not self.user_data.get("date"):
                self.pending_slot = "date"
                response = "When do you want to go?"
                self.chat_history.append("Bot: " + response)
                return response
            if not self.user_data.get("time"):
                self.pending_slot = "time"
                response = "What time do you want to go?"
                self.chat_history.append("Bot: " + response)
                return response
            if not self.user_data.get("number"):
                self.pending_slot = "number"
                response = "How many people will be going?"
                self.chat_history.append("Bot: " + response)
                return response

            response = self.nodes["booking"].get_response("book_restaurant")
            response = response.format(date=self.user_data["date"],
                                       time=self.user_data["time"],
                                       number=self.user_data["number"])
            self.chat_history.append("Bot: " + response)
            return response

        elif intent == "yes":
            # 'Yes' is interpreted as confirmation to start booking if details are missing.
            if not self.user_data.get("date"):
                self.pending_slot = "date"
                response = "When do you want to go?"
                self.chat_history.append("Bot: " + response)
                return response
            if not self.user_data.get("time"):
                self.pending_slot = "time"
                response = "What time do you want to go?"
                self.chat_history.append("Bot: " + response)
                return response
            if not self.user_data.get("number"):
                self.pending_slot = "number"
                response = "How many people will be going?"
                self.chat_history.append("Bot: " + response)
                return response
            response = self.nodes["booking"].get_response("book_restaurant")
            response = response.format(date=self.user_data["date"],
                                       time=self.user_data["time"],
                                       number=self.user_data["number"])
            self.chat_history.append("Bot: " + response)
            return response

        elif intent == "restaurant_opening_hours":
            response = self.nodes["opening_hours"].get_response("restaurant_opening_hours")
            followup = "Would you like to continue with your restaurant booking?"
            self.chat_history.append("Bot: " + response)
            self.chat_history.append("Bot: " + followup)
            return response + " " + followup

        elif intent == "job_opportunities":
            response = self.nodes["job"].get_response("job_opportunities")
            followup = "Would you like to continue with your restaurant booking?"
            self.chat_history.append("Bot: " + response)
            self.chat_history.append("Bot: " + followup)
            return response + " " + followup

        elif intent == "help":
            response = self.nodes["help"].get_response("help")
            followup = "Would you like to continue with your restaurant booking?"
            self.chat_history.append("Bot: " + response)
            self.chat_history.append("Bot: " + followup)
            return response + " " + followup

        elif intent == "cancel":
            response = self.nodes["cancel"].get_response("cancel")
            self.chat_history.append("Bot: " + response)
            self.user_data = {}
            self.pending_slot = None
            return response

        elif intent in ["greeting", "i_am_fine"]:
            response = self.nodes["greeting"].get_response(intent)
            followup = "Would you like to make a restaurant booking?"
            self.chat_history.append("Bot: " + response)
            self.chat_history.append("Bot: " + followup)
            return response + " " + followup

        elif intent == "goodbye":
            response = self.nodes["goodbye"].get_response("goodbye")
            self.chat_history.append("Bot: " + response)
            return response

        else:
            response = self.nodes["fallback"].get_response("fallback")
            self.chat_history.append("Bot: " + response)
            return response

# ---------------------------
# Node Initialization
# ---------------------------
bot = ChatBot()

# Booking node (frame response from JSON)
booking_node = ChatNode("booking")
booking_node.add_intent("book_restaurant", "OK. I'm making you a reservation on {date} at {time} for {number} people.")

# Digression nodes
opening_hours_node = ChatNode("opening_hours")
opening_hours_node.add_intent("restaurant_opening_hours", "The restaurant is open from 8:00 AM to 10:00 PM.")

job_node = ChatNode("job")
job_node.add_intent("job_opportunities", "We are always looking for talented people to add to our team. What type of job are you interested in?")

cancel_node = ChatNode("cancel")
cancel_node.add_intent("cancel", "Ok, cancelling the task. Your booking has been cancelled.")

help_node = ChatNode("help")
help_node.add_intent("help", "You can ask me to book a restaurant or inquire about our hours or job openings.")

greeting_node = ChatNode("greeting")
greeting_node.add_intent("greeting", "Hello. How can I help you?")
greeting_node.add_intent("i_am_fine", "Glad to hear that!")

fallback_node = ChatNode("fallback")
fallback_node.add_intent("fallback", "I didn't understand that. Could you please rephrase?")

goodbye_node = ChatNode("goodbye")
goodbye_node.add_intent("goodbye", "Goodbye! Have a great day.")

# Add all nodes to the chatbot.
bot.add_node(booking_node)
bot.add_node(opening_hours_node)
bot.add_node(job_node)
bot.add_node(cancel_node)
bot.add_node(help_node)
bot.add_node(greeting_node)
bot.add_node(fallback_node)
bot.add_node(goodbye_node)

@app.post("/api/data")
async def get_data(request: MessageRequest):
    user_input = request.message.strip()
    response = await bot.handle_input(user_input, request.use_gemini)
    print(f"Response: {response}")
    return {"message": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
