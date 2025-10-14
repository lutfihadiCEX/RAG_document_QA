# RAG Pipeline Module
# Combines retrieval and generation for question answering


from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from typing import Dict, List
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for document QA"""
    
    def __init__(self, vectorstore: FAISS, model_name: str = "llama3.2", temperature: float = 0.7):
        """
        Initialize RAG pipeline
        
        Args:
            vectorstore: FAISS vector store with indexed documents
            model_name: Ollama model to use (llama3.2, mistral, phi3)
            temperature: LLM temperature (0=deterministic, 1=creative)
        """
        self.vectorstore = vectorstore
        self.model_name = model_name
        
        # Initialize Ollama LLM
        logger.info(f"Initializing Ollama with model: {model_name}")
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434"  # Default Ollama port
        )
        
        # Custom prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Retrieval QA chain
        self.qa_chain = self._create_qa_chain()
        
        logger.info("RAG pipeline initialized successfully")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create a custom prompt template for the LLM"""
        template = """You are a helpful AI assistant answering questions based on provided documents.

Use the following context to answer the question accurately and concisely.
If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer (be specific and cite relevant parts of the context):"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create the retrieval QA chain"""
        # Configure retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3,  # Retrieve top 3 most relevant chunks
                "fetch_k": 10  # Fetch 10 candidates, return top 3
            }
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" = put all context in prompt
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True,
            verbose=False
        )
        
        return qa_chain
    
    def query(self, question: str) -> Dict:
        """
        Ask a question and get answer with sources
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"Processing query: {question[:100]}...")
        
        start_time = time.time()
        
        try:
            # Run the QA chain
            result = self.qa_chain({"query": question})
            
            # Extract answer and sources
            answer = result["result"]
            source_docs = result["source_documents"]
            
            # Calculate response time
            response_time = time.time() - start_time
            
            logger.info(f"Query completed in {response_time:.2f}s")
            
            return {
                "answer": answer,
                "sources": source_docs,
                "num_sources": len(source_docs),
                "response_time": response_time,
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "num_sources": 0,
                "response_time": time.time() - start_time,
                "model": self.model_name
            }
    
    def get_relevant_documents(self, question: str, k: int = 3) -> List:
        """
        Get relevant documents without generating an answer
        Useful for debugging retrieval
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)
        return docs
    
    def update_model(self, model_name: str):
        """Switch to a different Ollama model"""
        logger.info(f"Switching model from {self.model_name} to {model_name}")
        self.model_name = model_name
        self.llm = Ollama(model=model_name, temperature=self.llm.temperature)
        self.qa_chain = self._create_qa_chain()
        logger.info("Model updated successfully")


# Example usage
if __name__ == "__main__":
    from app.ingestion import DocumentIngestion
    
    # Load or create vector store
    ingestor = DocumentIngestion()
    
    try:
        # Try to load existing vector store
        vectorstore = ingestor.load_vectorstore()
        print("Loaded existing vector store")
    except FileNotFoundError:
        print("No existing vector store found. Please run ingestion first.")
        exit(1)
    
    # RAG pipeline
    rag = RAGPipeline(vectorstore, model_name="llama3.2")
    
    # Test query
    question = "What is this document about?"
    result = rag.query(question)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Response time: {result['response_time']:.2f}s")
    print(f"Sources used: {result['num_sources']}")