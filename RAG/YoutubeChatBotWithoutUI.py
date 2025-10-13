from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 1. Load YouTube Transcript
video_id = "RlGOKUFZqGo"
transcript = ''

try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id)
    transcript = " ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
    print("No captions available for this video.")

# 2. Chunk the transcript
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# 3. Generate embeddings using a Hugging Face model
embedding_model_id = "BAAI/bge-small-en"  # You can change to another HF embedding model
embeddings = HuggingFaceEndpointEmbeddings(repo_id=embedding_model_id)
vector_store = FAISS.from_documents(chunks, embeddings)

# 4. LLM setup using Hugging Face Endpoint
llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",  # Replace with any HF LLM you want
    task="text-generation",
    max_new_tokens=150,
    temperature=0.5
)

# Wrap it as a chat model (needed for LangChain)
chat_model = ChatHuggingFace(llm=llm_endpoint)

# 5. Create MultiQueryRetriever using Hugging Face LLM
retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(),
    llm=chat_model
)

# 6. Prompt template
prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY using the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""",
    input_variables=["context", "question"]
)

# 7. Chain construction
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# Prepare a parallel input for context and question
paralle_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

# Combine all parts into a chain
main_chain = paralle_chain | prompt | chat_model | StrOutputParser()

# 8. Invoke the chain with a query
query = "what is the video about?"
result = main_chain.invoke(query)
print(result)
