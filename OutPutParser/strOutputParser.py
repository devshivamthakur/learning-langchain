from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
    task="text-generation",
)

modal = ChatHuggingFace(llm = llm)

#1st prompt
prompt1 = PromptTemplate(
    template="write a details about {topic}",
    input_variables=['topic'],
)

#2nd prompt

prompt2 = PromptTemplate(
    template="write a 5 points on the following text.\n{text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | modal | parser | prompt2 | modal | parser

result = chain.invoke({
    "topic" : "react native"
})

print(result)