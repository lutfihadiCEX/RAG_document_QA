# Document Ingestion Module
# Loads, chunks, and indexes documents into vector store


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIngestion:
    """Loads PDFs, Word docs, or text files and builds a FAISS index for retrieval."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document ingestion pipeline
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks for context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Free sentence-transformers embeddings 
        logger.info("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Embedding model loaded successfully")
        
    def load_document(self, file_path: str):
        """
        Load document based on file extension
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of Document objects
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        logger.info(f"Loading document: {file_path}")
        
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages/sections from {os.path.basename(file_path)}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def process_documents(self, file_paths: List[str]) -> Tuple[FAISS, int]:
        """
        Process multiple documents into vector store
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            Tuple of (vectorstore, number of chunks)
        """
        all_docs = []
        
        # Load all documents
        for file_path in file_paths:
            try:
                docs = self.load_document(file_path)
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {str(e)}")
                continue
        
        if not all_docs:
            raise ValueError("No documents were successfully loaded")
        
        # Split into chunks
        logger.info(f"Splitting {len(all_docs)} documents into chunks...")
        chunks = self.text_splitter.split_documents(all_docs)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Create vector store
        logger.info("Creating vector store (this may take a minute)...")
        vectorstore = FAISS.from_documents(chunks, self.embeddings)
        logger.info("Vector store created successfully")
        
        return vectorstore, len(chunks)
    
    def save_vectorstore(self, vectorstore: FAISS, path: str = "data/vectorstore"):
        """Save vector store to disk"""
        os.makedirs(path, exist_ok=True)
        vectorstore.save_local(path)
        logger.info(f"Vector store saved to {path}")
    
    def load_vectorstore(self, path: str = "data/vectorstore") -> FAISS:
        """Load vector store from disk"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store not found at {path}")
        
        vectorstore = FAISS.load_local(
            path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Vector store loaded from {path}")
        return vectorstore
# Might try Chroma later for possible incremental updates

# Example usage
if __name__ == "__main__":
    # Test the ingestion pipeline
    ingestor = DocumentIngestion()
    
    # Test with sample documents
    test_files = ["data/documents/sample.pdf"]  # Replace with your files
    
    if os.path.exists(test_files[0]):
        vectorstore, num_chunks = ingestor.process_documents(test_files)
        print(f"Successfully processed {num_chunks} chunks")
        
        # Save for later use
        ingestor.save_vectorstore(vectorstore)
    else:
        print("Place test documents in data/documents/ folder")
