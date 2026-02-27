from dotenv import load_dotenv
from openai import OpenAI

import os

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_llm(user_prompt, system_prompt="You are a fat chud.", model="gpt-4.1-mini"):
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    return response.output_text

output = ask_llm("Deez nuts")
print(output)
