import streamlit as st
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

# Set page configuration
st.set_page_config(
    page_title="YouTube Video Q&A Assistant",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
        margin-top: 20px;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎬 YouTube Video Q&A Assistant</div>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Video ID input
    video_id = st.text_input(
        "YouTube Video ID:",
        value="Op6PbJZ5b2Q",
        help="The ID of the YouTube video (found in the URL after 'v=')"
    )
    
    # Model settings
    st.subheader("Model Settings")
    embedding_model = st.selectbox(
        "Embedding Model:",
        ["BAAI/bge-small-en", "sentence-transformers/all-mpnet-base-v2", "intfloat/e5-small-v2"],
        index=0
    )
    
    llm_model = st.selectbox(
        "LLM Model:",
        ["Qwen/Qwen3-Next-80B-A3B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "google/flan-t5-large"],
        index=0
    )
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        chunk_size = st.slider("Chunk Size:", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap:", 0, 500, 200, 50)
        max_tokens = st.slider("Max Response Tokens:", 50, 300, 150, 25)
        temperature = st.slider("Temperature:", 0.0, 1.0, 0.5, 0.1)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📺 Video Information")
    
    # Display video thumbnail and info
    if video_id:
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
        st.write(f"**Video ID:** {video_id}")
        st.markdown(f"[Watch on YouTube](https://www.youtube.com/watch?v={video_id})")

with col2:
    st.header("❓ Ask Questions")
    
    # Question input
    question = st.text_area(
        "Enter your question about the video:",
        value="what is the video about?",
        height=100,
        placeholder="Ask anything about the video content..."
    )
    
    # Process button
    process_btn = st.button("🚀 Process Video & Get Answer", type="primary", use_container_width=True)

# Processing and results section
if process_btn:
    if not video_id:
        st.error("❌ Please enter a YouTube Video ID")
        st.stop()
    
    if not question:
        st.error("❌ Please enter a question")
        st.stop()
    
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load YouTube Transcript
        status_text.text("📥 Loading YouTube transcript...")
        progress_bar.progress(10)
        
        transcript = ''
        try:
            youTubeObject = YouTubeTranscriptApi()
            tempTransList = youTubeObject.list(video_id)
            # Get the manually created or default transcript (typically the first one)
            # transcript_obj = tempTransList.find_manually_created() or tempTransList.find_generated_transcript(['en'])
            default_transcript = ''
            for t in tempTransList:
                if t.is_generated:
                    default_transcript = t
                    break

            isEnglishAvailable = any(t.language_code == 'en' for t in tempTransList)
            if isEnglishAvailable:
                transcript_obj = next(t for t in tempTransList if t.language_code == 'en')
            else:
                transcript_obj = default_transcript
            
            print(f"Using transcript in language: {transcript_obj.language_code}")
            #1.2 if transcript_obj.language_code != 'en' then convert to english using llm
            transcript = " ".join(chunk.text for chunk in youTubeObject.fetch(video_id, languages=[transcript_obj.language_code]))
            st.success(f"✅ Successfully loaded transcript ({len(transcript)} characters)")
        except TranscriptsDisabled:
            st.error("❌ No captions available for this video")
            st.stop()
        
        # Step 2: Chunk the transcript
        status_text.text("✂️ Chunking transcript...")
        progress_bar.progress(30)
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.create_documents([transcript])
        st.info(f"📚 Created {len(chunks)} chunks from the transcript")
        
        # Step 3: Generate embeddings
        status_text.text("🔤 Generating embeddings...")
        progress_bar.progress(50)
        
        embeddings = HuggingFaceEndpointEmbeddings(repo_id=embedding_model)
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Step 4: Setup LLM
        status_text.text("🤖 Initializing language model...")
        progress_bar.progress(70)
        
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=llm_model,
            task="text-generation",
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        chat_model = ChatHuggingFace(llm=llm_endpoint)
        
        # Step 5: Create retriever and chain
        status_text.text("🔗 Setting up Q&A system...")
        progress_bar.progress(85)
        
        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(),
            llm=chat_model
        )
        
        prompt = PromptTemplate(
            template="""
You are a helpful assistant that answers questions about YouTube video content.
Answer ONLY using the provided transcript context.
If the context is insufficient, just say you don't know.

Context from video transcript:
{context}

Question: {question}

Answer: """,
            input_variables=["context", "question"]
        )
        
        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        paralle_chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        
        main_chain = paralle_chain | prompt | chat_model | StrOutputParser()
        
        # Step 6: Get answer
        status_text.text("💭 Generating answer...")
        progress_bar.progress(95)
        
        result = main_chain.invoke(question)
        
        # Display results
        progress_bar.progress(100)
        status_text.text("✅ Complete!")
        
        st.header("🎯 Answer")
        st.markdown(f'<div class="response-box">{result}</div>', unsafe_allow_html=True)
        
        # Show transcript preview
        with st.expander("📄 View Transcript Preview"):
            st.text_area("First 1000 characters of transcript:", transcript[:1000] + "..." if len(transcript) > 1000 else transcript, height=200)
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.markdown('<div class="warning-box">Please check your video ID and try again. Make sure the video has captions enabled.</div>', unsafe_allow_html=True)

# Instructions section
with st.expander("📖 How to use this app"):
    st.markdown("""
    1. **Get YouTube Video ID**: 
       - Go to any YouTube video
       - Copy the ID from the URL (the part after `v=`)
       - Example: For `https://www.youtube.com/watch?v=abc123`, the ID is `abc123`
    
    2. **Configure Models** (optional):
       - Choose different embedding models for processing text
       - Select different language models for answering questions
       - Adjust advanced settings for better results
    
    3. **Ask Questions**:
       - Enter any question about the video content
       - Click "Process Video & Get Answer"
       - Wait for the AI to analyze the transcript and provide an answer
    
    **Note**: The video must have captions available for this to work.
    """)

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit, LangChain, and Hugging Face | "
    "Note: Processing may take some time depending on video length and model sizes"
)