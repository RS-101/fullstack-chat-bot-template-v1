from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import ollama

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Adjust to match your Vite app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a request model
class MessageRequest(BaseModel):
    message: str


class ChatNode:
    def __init__(self, name):
        self.name = name
        self.intents = {}

    def add_intent(self, intent, response):
        """Maps an intent to a response."""
        self.intents[intent] = response

    def get_response(self, intent):
        """Returns the response if the intent exists."""
        return self.intents.get(intent, None)

class ChatBot:
    def __init__(self):
        self.nodes = {}
        self.current_node = None

    def add_node(self, node):
        self.nodes[node.name] = node

    def set_start_node(self, node_name):
        """Sets the initial chatbot node."""
        self.current_node = self.nodes.get(node_name)

    async def classify_intent(self, user_input):
        """Uses an LLM to classify user input into an intent."""
        system_prompt = """
        You are an intent classifier. Given a user input, classify it into one of the following intents:
        - hello
        - help
        - goodbye
        - info
        - fallback (if it doesn't match any intent)
        
        Respond only with the intent name.
        """
        
        response = ollama.chat(
            model="gemma2:2b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
        )
        return response['message']['content'].strip().lower()

    async def handle_input(self, user_input):
        """Classifies user input and returns the appropriate response."""
        if not self.current_node:
            return "Error: No starting node set."

        # Get classified intent from LLM
        intent = await self.classify_intent(user_input)
        print(f"Intent: {intent}")
        # Handle recognized intent
        response = self.current_node.get_response(intent)
        if response:
            return response

        # Fallback to AI-generated response if no intent matches
        print(f"No predefined response for intent '{intent}'. Querying AI...")
        ai_response = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': user_input}])
        return ai_response['message']['content']


# --- Initialize Chatbot ---
bot = ChatBot()

# Create Nodes
greeting_node = ChatNode("greeting")
help_node = ChatNode("help")
goodbye_node = ChatNode("goodbye")

# Add intents and responses
greeting_node.add_intent("hello", "Hi there! How can I assist you?")
greeting_node.add_intent("help", "Sure, I can help! What do you need assistance with?")
greeting_node.add_intent("goodbye", "Goodbye! Have a great day.")

help_node.add_intent("info", "I can provide information on various topics. Ask me anything!")
help_node.add_intent("back", "Returning to greeting...")

goodbye_node.add_intent("bye", "Farewell! See you next time.")

# Add Nodes to Bot
bot.add_node(greeting_node)
bot.add_node(help_node)
bot.add_node(goodbye_node)

# Set the Start Node
bot.set_start_node("greeting")


@app.post("/api/data")
async def get_data(request: MessageRequest):
    """Handles chatbot requests, classifies intent, and responds accordingly."""
    
    user_input = request.message.strip()
    response = await bot.handle_input(user_input)

    print(f"Response: {response}")
    return {"message": response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
