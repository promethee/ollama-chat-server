#!/usr/bin/python
import logging
import uvicorn
import signal
from uvicorn.config import LOGGING_CONFIG
import requests
import json
from enum import Enum, IntEnum
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ollama import Client

LOGGING_CONFIG['formatters']['default']['fmt'] = '%(levelprefix)s %(message)s'
logger = logging.getLogger('uvicorn.error')
logger.name = 'uvicorn'
logger.info("")
logger.info("STARTING")

app = FastAPI()

OLLAMA_URL = "http://localhost:11434"
PORT = 11435
CORS_ALLOW_ORIGINS = ['*']

try:
    with open('settings.json', 'r') as file:
        settings = json.load(file)
        logger.info("settings.json file found")

        if "ollama_url" in settings:
            OLLAMA_URL = settings["ollama_url"]
            logger.info("\tusing OLLAMA_URL = " + settings["ollama_url"])
            
        if "host" in settings:
            HOST = settings["host"]
            logger.info("\nusing HOST = " + settings["host"])

        if "port" in settings:
            PORT = settings["port"]
            logger.info("\tusing PORT = " + str(settings["port"]))

        if "allow_origins" in settings:
            CORS_ALLOW_ORIGINS = settings["allow_origins"]
            logger.info("\tusing CORS_ALLOW_ORIGINS = " + str(settings["allow_origins"]))
    
except FileNotFoundError:
    logger.info("settings.json file not found, using default values")

client = Client(host=OLLAMA_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins = CORS_ALLOW_ORIGINS,
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

def stream_ollama_response(payload):
    url = OLLAMA_URL + '/api/chat'
    with requests.post(url, json=payload.model_dump(), stream=True) as response:
        if response.status_code == 200:
            line_count = 0
            for line in response.iter_lines():
                if line:
                    line_count += 1
                    logger.info("\tReturning chunk {count} to client".format(count=line_count))
                    yield line.decode("utf-8") + "\n"
        else:
            return response.json()

@app.post("/api/chat")
async def _api_chat(payload: ChatPayload):
    logger.info("")
    logger.info("Sending user message to Ollama model")
    if payload.stream:
        logger.info("Streaming the response, chunk by chunk")
        return StreamingResponse(stream_ollama_response(payload), media_type="application/json")
    else:
        logger.info("Not streaming the response, waiting for it")
        response = client.chat(model=payload.model, messages=payload.messages, stream=payload.stream)
        logger.info("Sending the response")
        return response

@app.get("/api/tags")
async def _api_tags():
    global models, model
    logger.info("")
    logger.info("Returning models list")
    return client.list()

@app.get("/api")
async def root():
    return "https://github.com/ollama/ollama/blob/main/docs/api.md"

app.mount("/", StaticFiles(directory="client", html=True), name="client")

if __name__ == "__main__":
    logger.info("Ollama Client @ " + OLLAMA_URL)
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True, log_level="debug")
