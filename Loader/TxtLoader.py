from langchain.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

#load the text file
loader = TextLoader("sample.txt", encoding='utf-8')
documents = loader.load()

from pydantic import BaseModel, Field

class TextSummary(BaseModel):
    summary: str = Field(..., description="A concise summary of the provided text.")
    key_points: list[str] = Field(..., description="A list of 5 key points from the text.")
    topic: str = Field(..., description="The main topic of the text. Must be a single word")

parser = PydanticOutputParser(pydantic_object=TextSummary)

prompt = PromptTemplate(
    template="Summarize the following text:\n{text}\n${format}",
    input_variables=["text"],
    partial_variables={
        "format": parser.get_format_instructions()
    }
)


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task='text-generation',
    max_new_tokens=150,
    temperature=0.5
) # pyright: ignore[reportCallIssue]

model = ChatHuggingFace(llm=llm)
chain =  prompt | model | parser

result = chain.invoke({"text": documents[0].page_content})

print(result.model_dump_json())


