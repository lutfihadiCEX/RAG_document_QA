# Streamlit UI for RAG Document QA System


import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion import DocumentIngestion
from app.pipeline import RAGPipeline
import tempfile

# Page configuration
st.set_page_config(
    page_title="RAG Document QA",
    page_icon="🧠🛢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🧠🛢 RAG Document Q&A System</p>', unsafe_allow_html=True)
st.markdown("*Upload documents and ask questions - Powered by local LLMs (100% FREE!)*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model = st.selectbox(
        "Select LLM Model",
        ["llama3.2", "mistral", "phi3"],
        help="Choose which Ollama model to use"
    )
    
    st.info(f"Using: **{model}**")
    
    st.divider()
    
    # Upload section
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload one or more documents to analyze"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        for file in uploaded_files:
            st.text(f"📄 {file.name}")
    
    st.divider()
    
    # Processing settings
    with st.expander("🔧 Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
        temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.7, 0.1)
        num_sources = st.slider("Sources to Retrieve", 1, 5, 3, 1)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Document processing
    if uploaded_files and st.button("🔄 Process Documents", type="primary", use_container_width=True):
        with st.spinner("Processing documents... This may take a minute ⏳"):
            try:
                # Save uploaded files temporarily
                temp_dir = tempfile.mkdtemp()
                file_paths = []
                
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    file_paths.append(file_path)
                
                # Initialize ingestion with custom settings
                ingestor = DocumentIngestion(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Process documents
                vectorstore, num_chunks = ingestor.process_documents(file_paths)
                
                # Save vector store
                ingestor.save_vectorstore(vectorstore)
                
                # Create RAG pipeline
                st.session_state.vectorstore = vectorstore
                st.session_state.rag_pipeline = RAGPipeline(
                    vectorstore, 
                    model_name=model,
                    temperature=temperature
                )
                st.session_state.processed_files = [f.name for f in uploaded_files]
                
                st.success(f"✅ Successfully processed {len(file_paths)} documents into {num_chunks} chunks!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")
                st.info("Make sure Ollama is running: `ollama serve`")

with col2:
    # Statistics
    if st.session_state.vectorstore:
        st.metric("Documents Processed", len(st.session_state.processed_files))
        st.metric("Model", model)
        st.metric("Status", "✅ Ready")
    else:
        st.info("⬅️ Upload and process documents to get started")

# Main Q&A section: This is where user asks questions after docs are processed
if st.session_state.rag_pipeline:
    st.divider()
    st.header("💭 Ask Questions")
    
    # Chat interface
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the main topic of the documents?",
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        ask_button = st.button("💡 Ask", type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear History")
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if ask_button and question:
        with st.spinner("🧐 Thinking..."):
            try:
                # Update model if changed
                if st.session_state.rag_pipeline.model_name != model:
                    st.session_state.rag_pipeline.update_model(model)
                
                # Query the RAG pipeline
                result = st.session_state.rag_pipeline.query(question)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "response_time": result["response_time"]
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Make sure Ollama is running: `ollama serve`")
    
    # Display chat history 
    if st.session_state.chat_history:
        st.divider()
        st.subheader("📖 Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question'][:100]}...", expanded=(i==0)):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                st.caption(f"⏳ Response time: {chat['response_time']:.2f}s")
                
                # Show sources
                if chat['sources']:
                    st.markdown("**📄 Sources:**")
                    for j, doc in enumerate(chat['sources'][:3], 1):
                        with st.container():
                            st.markdown(f"<div class='source-box'>", unsafe_allow_html=True)
                            st.markdown(f"**Source {j}:**")
                            st.text(doc.page_content[:300] + "...")
                            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                                st.caption(f"📎 From: {os.path.basename(doc.metadata['source'])}")
                            st.markdown("</div>", unsafe_allow_html=True)

else:
    # Welcome message
    st.info("""
    ### 👋🤗 Welcome to RAG Document QA!
    
    **How to use:**
    1. Upload documents (PDF, TXT, DOCX) in the sidebar
    2. Click "Process Documents"
    3. Ask questions about your documents
    
    **Example questions:**
    - "What is the main topic of these documents?"
    - "Summarize the key findings"
    - "What are the main recommendations?"
    
    **Note:** Make sure Ollama is running (`ollama serve`)
    """)

# Footer
st.divider()
st.caption("🚀 Built with LangChain, Ollama, FAISS & Streamlit | 100% Local & Free")
