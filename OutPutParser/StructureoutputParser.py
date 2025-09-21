from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

modal = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(name = "Fact_1", description="fact 1 about the topic"),
    ResponseSchema(name = "Fact_2", description="fact 2 about the topic"),
    ResponseSchema(name = "Fact_3", description="fact 3 about the topic"),
    ResponseSchema(name = "Fact_4", description="fact 4 about the topic"),
    ResponseSchema(name = "Fact_5", description="fact 5 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='give me 5 fact about react native in the {format_ins}',
    input_variables=[],
    partial_variables={'format_ins': parser.get_format_instructions()}
)

chain = template | modal | parser

result = chain.invoke({})

print(result['Fact_1'])