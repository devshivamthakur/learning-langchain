from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

prompt = PromptTemplate(
    template="What is a good name for a company that makes {product}?",
    input_variables=["product"]
)

formattedPromt = prompt.invoke({"product" : "smart watch"})

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest', temperature=0.7, max_tokens=150)
result = llm.invoke(formattedPromt)

print(result.content)


