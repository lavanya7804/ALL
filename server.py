import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re


genai.configure(api_key="AIzaSyB9JqTbiIBxydSxbds0Y2qUcXiEnyLQGM4")


generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
    system_instruction="You are a friendly chatbot that provides helpful responses."
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


history = []


class MessageRequest(BaseModel):
    user_input: str

@app.post("/chat/")
async def chat(request: MessageRequest):
    global history

    try:
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(request.user_input)
      
        if response.text:
            model_response = response.text
            model_response = re.sub(r'^\+\s', '', model_response, flags=re.MULTILINE)  
            model_response = re.sub(r'\\(.)', r'\1', model_response)  
            model_response = re.sub(r'\*+', '', model_response)  

        else:
            model_response = "Sorry, I couldn't process your request."

        history.append({"role": "user", "parts": [{"text": request.user_input}]})
        history.append({"role": "model", "parts": [{"text": model_response}]})

        return {"response": model_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def home():
    return {"message": "LLM Chatbot Backend Running"}


    