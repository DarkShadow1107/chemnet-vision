import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def process_pdfs():
    pdf_dir = os.path.join('data', 'pdfs')
    if not os.path.exists(pdf_dir):
        print(f"PDF directory not found: {pdf_dir}")
        return

    documents = []
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found to process.")
        return

    print(f"Found {len(pdf_files)} PDFs. Loading...")
    
    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_dir, pdf_file)
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            # Add metadata to help with retrieval if needed
            for doc in docs:
                doc.metadata['source'] = pdf_file
            documents.extend(docs)
            print(f"Loaded {pdf_file}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

    if not documents:
        print("No documents loaded.")
        return

    print(f"Total pages loaded: {len(documents)}")

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # Save chunks to JSON
    import json
    chunks_data = [{'content': c.page_content, 'metadata': c.metadata} for c in chunks]
    with open(os.path.join('data', 'chunks.json'), 'w') as f:
        json.dump(chunks_data, f, indent=4)
    print("Saved chunks to data/chunks.json")

    # Create Embeddings
    print("Creating embeddings (this may take a while)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Save embeddings to NPY
    import numpy as np
    # We need to embed the documents to get the vectors for saving
    # Note: FAISS.from_documents also does this, but we do it here to save explicitly
    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    np.save(os.path.join('data', 'embeddings.npy'), np.array(vectors))
    print("Saved embeddings to data/embeddings.npy")

    # Create Vector Store
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save
    save_path = os.path.join('data', 'faiss_index')
    vector_store.save_local(save_path)
    print(f"FAISS index saved to {save_path}")

if __name__ == "__main__":
    process_pdfs()
