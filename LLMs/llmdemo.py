from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest', temperature=0.7)
result = llm.invoke("how to seach better in any gpt modal")
print(result.content)
