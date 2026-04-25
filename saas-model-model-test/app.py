from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI(title="Saas Companies Information API")

OLLAMA_URL = "http://localhost:11434/api/generate"

class Query(BaseModel):
    input: str

@app.post("/advice")
async def get_advice(query: Query):
    prompt = f"""
    Input: {query.input}
    Response:
    """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "saas-orgs:latest",  # ✅ your model name here
            "prompt": prompt,
            "stream": False,
        }
    )

    result = response.json()
    # cleaned_text = result["response"].replace("<think>\n", "").replace("\n</think>\n", "")
    return {"data": result["response"]}