# ingest.py
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("data")                 # 放 PDF 的資料夾
DB_DIR = Path("chroma_db")              # Chroma 持久化目錄
COLLECTION = "my-collection"            # 向量庫名稱

def main():
    # 1) DocumentLoader
    loader = PyPDFDirectoryLoader(str(DATA_DIR))
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")


    # 2) TextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？"]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # 3) Embeddings
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # 4) VectorStore (Chroma)
    vs = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding,
        persist_directory=str(DB_DIR)
    )

    # 清一次（可選）：避免重複累積
    vs.delete_collection()
    vs = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding,
        persist_directory=str(DB_DIR)
    )

    # 寫入
    vs.add_documents(chunks)

    # Fetch all docs from vector store
    docs = vs.similarity_search("", k=len(chunks))  # empty query just retrieves stored docs

    for i, doc in enumerate(docs, 1):
        print(f"--- Document {i} ---")
        print("Content:", doc.page_content[:200], "...")
        print("Metadata:", doc.metadata)
        print()

    print("✅ Vector store built & persisted.")

if __name__ == "__main__":
    main()
