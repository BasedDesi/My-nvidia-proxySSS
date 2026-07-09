from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import os
import uvicorn
import json
import sys

app = FastAPI()

# Настройка CORS с поддержкой всех методов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Инициализация клиента NVIDIA
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY environment variable is not set")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

# Модель для запроса от JanitorAI
class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 1.0
    max_tokens: int = 16384
    stream: bool = False
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

@app.options("/v1/chat/completions")
async def options_chat():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.options("/{path:path}")
async def options_all(path: str):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        print("=" * 50)
        print(f"📥 Получен запрос для модели: {request.model}")
        print(f"📝 Сообщений: {len(request.messages)}")
        
        # Базовые параметры для NVIDIA
        params = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream
        }
        
        # Добавляем опциональные параметры
        if request.top_p != 1.0:
            params["top_p"] = request.top_p
        if request.frequency_penalty != 0.0:
            params["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty != 0.0:
            params["presence_penalty"] = request.presence_penalty
        
        # 🔥 ВКЛЮЧАЕМ REASONING ДЛЯ GLM-5.2
        if "glm-5.2" in request.model.lower():
            print("🧠 Активация reasoning для GLM-5.2")
            params["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": True
                }
            }
        
        print("🔄 Отправка запроса в NVIDIA...")
        completion = client.chat.completions.create(**params)
        print("✅ Ответ от NVIDIA получен")
        
        # Проверяем, есть ли reasoning в ответе
        has_reasoning = hasattr(completion.choices[0].message, 'reasoning_content')
        if has_reasoning:
            reasoning = completion.choices[0].message.reasoning_content
            print(f"🧠 Reasoning найден! Длина: {len(reasoning)} символов")
            print(f"🧠 Содержание: {reasoning[:200]}...")
        else:
            print("❌ Reasoning ОТСУТСТВУЕТ в ответе NVIDIA")
        
        # Обработка стриминга
        if request.stream:
            def generate():
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta
                        response_data = {"choices": [{"delta": {}}]}
                        
                        if hasattr(delta, 'content') and delta.content:
                            response_data["choices"][0]["delta"]["content"] = delta.content
                        
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                            print("🧠 Reasoning в стриминге")
                            response_data["choices"][0]["delta"]["reasoning_content"] = delta.reasoning_content
                        
                        yield f"data: {json.dumps(response_data)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        # Обычный ответ
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": completion.choices[0].message.content
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        if has_reasoning:
            response_data["choices"][0]["message"]["reasoning_content"] = completion.choices[0].message.reasoning_content
        
        print("📤 Отправка ответа клиенту")
        return JSONResponse(response_data)
        
    except Exception as e:
        print(f"❌ ОШИБКА: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "api_error"
                }
            }
        )

@app.get("/")
def root():
    return {"status": "ok", "message": "NVIDIA Proxy for JanitorAI"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Запуск сервера на порту {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
