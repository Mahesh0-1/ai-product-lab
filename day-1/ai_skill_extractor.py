import os
import json
from openai import OpenAI

if len(job_description.strip()) < 10:
    raise ValueError("Job description too short.")

def get_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not set.")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )


def extract_skills(job_description):
    client = get_client()

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """
                Extract only technical skills.
                Return response in valid JSON format like:
                {
                  "skills": ["skill1", "skill2", "skill3"]
                }
                """
            },
            {
                "role": "user",
                "content": job_description
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Model did not return valid JSON", "raw_output": content}


if __name__ == "__main__":
    job_description = input("Enter job description:\n")

    result = extract_skills(job_description)

    print("\n🔹 Structured Output:\n")
    print(json.dumps(result, indent=2))