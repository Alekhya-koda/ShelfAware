from typing import Dict, List, Optional
import openai
import os

class ChatbotService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        
        self.emotions = [
            "happy", "sad", "angry", "excited", "scared", "romantic", 
            "suspenseful", "dark", "hopeful", "nostalgic", "peaceful", 
            "curious", "empowered", "lonely", "grateful", "confused", 
            "inspired", "amused", "moved", "adventurous", "reflective", 
            "whimsical", "heartbroken", "triumphant"
        ]
    
    def detect_mood(self, message: str) -> List[str]:
        message_lower = message.lower()
        
        mood_keywords = {
            "happy": ["happy", "joy", "great", "wonderful"],
            "sad": ["sad", "depressed", "down", "unhappy", "lonely"],
            "angry": ["angry", "mad", "frustrated", "annoyed"],
            "excited": ["excited", "thrilled", "pumped"],
            "romantic": ["love", "romantic", "romance"],
            "adventurous": ["adventure", "exciting", "thrilling"]
        }
        
        for mood, keywords in mood_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return [mood]
        
        return ["peaceful"]
    
    def generate_response(self, mood: str) -> str:
        responses = {
            "happy": "That's wonderful! Here are some joyful reads:",
            "sad": "I understand you're feeling down. Here are some uplifting books:",
            "angry": "I hear you. These books might help process those feelings:",
            "excited": "Love the energy! These thrilling reads match your vibe:",
            "scared": "Looking for something that captures that feeling? Try these:",
            "romantic": "Looking for romance? I have lovely suggestions:",
            "suspenseful": "Want something gripping? These will keep you on edge:",
            "dark": "In the mood for something intense? These are perfect:",
            "hopeful": "Here are some inspiring, hopeful stories:",
            "nostalgic": "These books will take you down memory lane:",
            "peaceful": "Here are some calming, peaceful reads:",
            "curious": "These fascinating books will satisfy your curiosity:",
            "empowered": "Here are some empowering, strong stories:",
            "lonely": "I understand. These books offer comfort and connection:",
            "grateful": "Here are some heartwarming, appreciative reads:",
            "confused": "These thought-provoking books might help:",
            "inspired": "Here are some inspirational reads:",
            "amused": "Want something fun? These will make you laugh:",
            "moved": "Here are some deeply moving stories:",
            "adventurous": "Ready for adventure? These will take you on a journey:",
            "reflective": "These contemplative books are perfect for reflection:",
            "whimsical": "Here are some delightfully whimsical reads:",
            "heartbroken": "I'm sorry you're hurting. These books offer solace:",
            "triumphant": "Celebrate with these triumphant stories:"
        }
        return responses.get(mood, "Here are some books you might enjoy:")
    
    def process_message(self, message: str, user_id: Optional[str] = None) -> Dict:
        moods = self.detect_mood(message)
        primary_mood = moods[0]
        
        response_text = self.generate_response(primary_mood)
        
        books = []
        
        follow_ups = [
            "Would you like books in a different mood?",
            "Tell me more about what you're looking for",
            "Want more recommendations?"
        ]
        
        return {
            "response": response_text,
            "mood": primary_mood,
            "books": books,
            "follow_up_questions": follow_ups
        }
