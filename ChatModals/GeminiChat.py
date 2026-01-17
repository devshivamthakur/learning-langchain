from langchain_openai import ChatOpenAI
from os import getenv
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1",
    model="google/gemma-3-4b-it:free",
)

# Example usage
response = llm.invoke("Who is the pm of india?")
print(response.content)
