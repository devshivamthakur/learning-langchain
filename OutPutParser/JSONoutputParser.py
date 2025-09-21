from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

modal = ChatHuggingFace(llm = llm)
parser = JsonOutputParser()

template = PromptTemplate(
    template='give me 5 fact about react native in the {format_ins}',
    input_variables=[],
    partial_variables={'format_ins': parser.get_format_instructions()}
)

chain = template | modal | parser

result = chain.invoke({})

print(result)