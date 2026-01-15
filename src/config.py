import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DB_PATH = os.path.join(BASE_DIR, "qdrant_db")
STORE_PATH = os.path.join(BASE_DIR, "parent_store")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

COLLECTION_NAME = "rag_collection"
DENSE_MODEL = "sentence-transformers/all-mpnet-base-v2"
SPARSE_MODEL = "Qdrant/bm25"
LLM_MODEL = "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:IQ3_XXS" 

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
PARENT_CHUNK_SIZE = 2000