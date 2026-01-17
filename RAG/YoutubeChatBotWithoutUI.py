from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from langchain_classic.retrievers import BM25Retriever,EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers import EnsembleRetriever
import os

load_dotenv()
base_url = os.getenv("base_url")
chat_model = os.getenv("chat_model")
embedding_model = os.getenv("embedding_model")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# 1. Load YouTube Transcript
video_id = "oSopCFhtB9s"
transcript = ''

#load transcript in default language or english if available
try:
        yt = YouTubeTranscriptApi()
        print("Fetching transcript...", flush=True)
        trans_list = yt.list(video_id)
        default = next(t for t in trans_list if t.is_generated)
        is_english = any(t.language_code == "en" for t in trans_list)
        transcript_obj = (
            next(t for t in trans_list if t.language_code == "en")
            if is_english
            else default
        )

        # Fetch the transcript text based on the defautt or english transcript object
        fetched = yt.fetch(video_id, languages=[transcript_obj.language_code])
        transcript = " ".join(x.text for x in fetched)

except TranscriptsDisabled:
    print("No captions available for this video.")

# 2. Chunk the transcript
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# 3. Generate embeddings using a Hugging Face model
embedding_model_id = "mixedbread-ai/mxbai-embed-large-v1"  # You can change to another HF embedding model
embeddings = HuggingFaceEndpointEmbeddings(repo_id=embedding_model_id, huggingfacehub_api_token=os.getenv("HUGGING_FACE_TOKEN"))
vector_store = FAISS.from_documents(chunks, embeddings) # semantic vector store
bm25_retriever = BM25Retriever.from_documents(chunks) # Keyword-based retriever
bm25_retriever.k = 4

chat_model = ChatOpenAI(
    api_key=openrouter_api_key,
    base_url=base_url,
    model=chat_model,
    temperature=0.2,
    streaming=True,
)

faissRetriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25})
# 5. Retriever setup for hybrid search
retriever = EnsembleRetriever(
    retrievers=[faissRetriever, bm25_retriever],
    weights=[0.7, 0.3]
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
