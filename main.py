from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# NVIDIA Client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 1.0
    max_tokens: int = 16384
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        completion = client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream
        )
        
        if request.stream:
            return StreamingResponse(stream_generator(completion), media_type="text/event-stream")
        
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": completion.choices[0].message.content
                }
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def stream_generator(completion):
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content:
            yield f"data: {{\"choices\": [{{\"delta\": {{\"content\": \"{chunk.choices[0].delta.content}\"}}}}]}}\n\n"
    yield "data: [DONE]\n\n"

# Для локального запуска
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
