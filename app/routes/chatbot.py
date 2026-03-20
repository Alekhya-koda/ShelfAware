
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List, Dict
from app.services.chatbot_service import ChatbotService

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    mood: str
    books: List[Dict]
    follow_up_questions: List[str]

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    chatbot = ChatbotService()
    result = chatbot.process_message(request.message, request.user_id)
    return result
