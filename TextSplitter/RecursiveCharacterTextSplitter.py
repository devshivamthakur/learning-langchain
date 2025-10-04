from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """LangChain is a framework that helps developers build applications
that are powered by language models. It provides tools to chain prompts,
models, and data sources together..."""

# Create a text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,      # max characters per chunk
    chunk_overlap=20,    # overlap between chunks to maintain context
)

# Split text
chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")
