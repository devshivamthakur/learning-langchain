from langchain_text_splitters import CharacterTextSplitter

document = """LangChain is a framework for developing applications powered by language models. It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more.
"""
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=5, chunk_overlap=2
)
texts = text_splitter.split_text(document)

print(texts)