from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-Next-80B-A3B-Instruct',
    task='text-generation',
    provider="auto",  # let Hugging Face choose the best provider for you
    max_new_tokens=150,
    temperature=0.5
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("what is capital of india?")
print(result.content)