#!/usr/bin/python
import requests
import json
from enum import Enum, IntEnum
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ollama import Client

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_credentials=True,
    allow_methods = ['*'],
    allow_headers = ['*']
)

class MessageRole(str, Enum):
    user = 'user'
    assistant = 'assistant'
    system = 'system'

class Message(BaseModel):
    role: MessageRole
    content: str

class ChatPayload(BaseModel):
    messages: list[Message]
    model: str
    stream: bool

OLLAMA_URL = "http://192.168.0.66:11434"

client = Client(
    host=OLLAMA_URL
)

def stream_ollama_response(payload):
    url = OLLAMA_URL + '/api/chat'
    with requests.post(url, json=payload.model_dump(), stream=True) as response:
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    yield line.decode("utf-8") + "\n"
        else:
            return response.json()

@app.post("/api/chat")
async def _api_chat(payload: ChatPayload):
    if payload.stream:
        return StreamingResponse(stream_ollama_response(payload), media_type="application/json")
    else:
        response = client.chat(model=payload.model, messages=payload.messages, stream=payload.stream)
        return response

@app.get("/api/tags")
async def _api_tags():
    return client.list()

@app.get("/")
async def root():
    return "https://github.com/ollama/ollama/blob/main/docs/api.md"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=11434, reload=True, log_level="debug")