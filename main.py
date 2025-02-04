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

@app.post("/api/data")
async def get_data(request: MessageRequest):
    # Log received message
    print(f"Received message: {request.message}")
    
    response = ollama.chat(model='deepseek-r1:7b', messages=[{'role': 'user', 'content': request.message}])

    # Print the response
    print(response['message']['content'])

    # Return a response
    return {"message": response['message']['content']}










if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
