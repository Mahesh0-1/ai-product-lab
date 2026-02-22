import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()


class JobRequest(BaseModel):
    job_description: str


def get_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not set.")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )


@app.post("/extract-skills")
def extract_skills(request: JobRequest):
    if len(request.job_description.strip()) < 10:
        return {"error": "Job description too short."}

    client = get_client()

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """
                Extract only technical skills.
                Return response in valid JSON format:
                {
                  "skills": ["skill1", "skill2"]
                }
                """
            },
            {
                "role": "user",
                "content": request.job_description
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON from model", "raw_output": content}