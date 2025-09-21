from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model_name="claude-3-5-haiku-latest")
result = llm.invoke("how to select any modal")
print(result.content)