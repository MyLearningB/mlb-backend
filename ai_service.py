import json
import uuid
import os 
from dotenv import load_dotenv
from datetime import date, datetime, timedelta
from typing import Optional
from openai import AsyncOpenAI

# This tells Python to look for the .env file and load it
load_dotenv()

# Initialize the client pointing to DeepSeek's API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here")

client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY, 
    base_url="https://api.deepseek.com" # Overriding the base URL to DeepSeek
)

# ==========================================
# 1. AI SOLVE ENGINE (Screen 10)
# ==========================================
async def generate_deepseek_solution(question_text: str) -> dict:
    """
    Sends the user's question to DeepSeek and forces it to return 
    a highly structured JSON response matching our Flutter app's expectations.
    """
    
    system_prompt = """
    You are myLB AI, an expert academic tutor.
    Analyze the user's question and provide a clear, step-by-step solution.
    
    You MUST respond in valid JSON format exactly matching this structure:
    {
      "steps": [
        {
          "step_number": 1,
          "text": "Your detailed explanation for this step...",
          "highlight_terms": [
            {"term": "exact word to highlight", "color": "mint"}
          ]
        }
      ],
      "confidence_score": 0.95
    }
    
    Rules for highlight_terms:
    - Use 'mint' for positive concepts, formulas, or correct answers.
    - Use 'peach' for warnings, common mistakes, or critical exceptions.
    - Do not highlight whole sentences, only key terms.
    - If there are no terms to highlight, leave the array empty [].
    """

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Solve this: {question_text}"}
            ],
            response_format={"type": "json_object"} 
        )

        ai_data = json.loads(response.choices[0].message.content)
        
        ai_data["solution_id"] = f"sol_{uuid.uuid4().hex[:8]}"
        ai_data["canvas_links"] = [] 
        
        if "confidence_score" not in ai_data:
            ai_data["confidence_score"] = 0.90
            
        return ai_data

    except Exception as e:
        print(f"DeepSeek AI Solve Error: {e}")
        return None

# ==========================================
# 2. AI STUDY PLAN GENERATOR (Screen 9)
# ==========================================
async def generate_deepseek_study_plan(
    goal: str, 
    target_date: Optional[date] = None, 
    days_remaining: Optional[int] = None
) -> dict:
    """
    Prompts DeepSeek to generate a structured 7-day study schedule.
    Adapts dynamically for strict deadlines vs continuous learning.
    """
    today_str = date.today().isoformat()
    
    # Dynamic Prompt Logic
    if target_date and days_remaining is not None:
        context_prompt = f"Their deadline is in {days_remaining} days ({target_date}). Focus on preparing them for this specific target, escalating difficulty appropriately."
        stats_days = days_remaining
    else:
        context_prompt = "They are learning continuously without a strict deadline. Focus on building a sustainable, consistent foundational routine."
        stats_days = -1 # Indicates no deadline to the frontend
    
    system_prompt = f"""
    You are myLB AI, an expert academic planner. 
    The user wants to study for "{goal}". {context_prompt}
    Today is {today_str}.
    
    Generate a highly realistic, balanced 7-day study plan starting from today.
    Include rest days to prevent burnout.
    
    You MUST respond in valid JSON format exactly matching this structure:
    {{
      "stats": {{
        "days_remaining": {stats_days},
        "daily_target_mins": 60,
        "topics_count": 5
      }},
      "week": [
        {{
          "date": "YYYY-MM-DD",
          "day_label": "MON", 
          "has_session": true,
          "session_type": "study"
        }}
      ],
      "sessions": [
        {{
          "date": "YYYY-MM-DD",
          "time": "16:00",
          "subject": "Specific topic name",
          "duration_mins": 30,
          "mode": "flashcard",
          "priority": "normal"
        }}
      ],
      "nudge": null 
    }}
    
    Rules for 'week': Generate EXACTLY 7 items for the next 7 days. session_type must be "study", "review", or "rest".
    Rules for 'sessions': Generate ONLY the sessions for days where has_session is true. mode must be "flashcard", "feynman", or "review". priority must be "normal", "high", or "weak_area".
    """

    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate my 7-day study plan for {goal}."}
            ],
            response_format={"type": "json_object"} 
        )

        plan_data = json.loads(response.choices[0].message.content)
        return plan_data

    except Exception as e:
        print(f"DeepSeek AI Plan Error: {e}")
        return None