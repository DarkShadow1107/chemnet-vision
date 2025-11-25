import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class RAGSystem:
    def __init__(self, index_path=None):
        if index_path is None:
            self.index_path = os.path.join('data', 'faiss_index')
        else:
            self.index_path = index_path
            
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True # Local file, safe
                )
                print("RAG Index loaded successfully.")
            except Exception as e:
                print(f"Error loading RAG index: {e}")
        else:
            print("RAG Index not found. Please run scripts/process_pdfs_for_rag.py")

    def query(self, text, k=3):
        if not self.vector_store:
            return "Knowledge base not available."
        
        try:
            docs = self.vector_store.similarity_search(text, k=k)
            # Combine content
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            return f"Error querying knowledge base: {e}"

# Global instance
rag_system = RAGSystem()

def get_rag_context(query_text):
    return rag_system.query(query_text)
