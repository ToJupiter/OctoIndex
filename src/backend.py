import json
import shutil
import uuid
import glob
from pathlib import Path
from typing import List, Literal, Annotated
import pymupdf
import pymupdf4llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from qdrant_client import QdrantClient, models as qmodels
import config

class Database:
    def __init__(self):
        self.client = QdrantClient(path=config.DB_PATH)
        self.dense_embed = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)
        self.sparse_embed = FastEmbedSparse(model_name=config.SPARSE_MODEL)
        self.store_path = Path(config.STORE_PATH)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._init_collection()

    def _init_collection(self):
        if not self.client.collection_exists(config.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(size=768, distance=qmodels.Distance.COSINE),
                sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
            )

    def get_vector_store(self):
        return QdrantVectorStore(
            client=self.client,
            collection_name=config.COLLECTION_NAME,
            embedding=self.dense_embed,
            sparse_embedding=self.sparse_embed,
            sparse_vector_name="sparse",
            retrieval_mode=RetrievalMode.HYBRID
        )

    def save_parent(self, pid, content, meta):
        (self.store_path / f"{pid}.json").write_text(
            json.dumps({"content": content, "metadata": meta}, ensure_ascii=False), encoding="utf-8"
        )

    def load_parent(self, pid):
        try:
            data = json.loads((self.store_path / f"{pid}.json").read_text(encoding="utf-8"))
            return f"Source: {data['metadata']['source']}\nContent: {data['content']}"
        except Exception as e:
            return f"Error loading document: {str(e)}"

    def clear(self):
        if self.store_path.exists(): shutil.rmtree(self.store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.client.delete_collection(config.COLLECTION_NAME)
        self._init_collection()

class Ingestor:
    def __init__(self, db: Database):
        self.db = db
        self.vector_store = db.get_vector_store()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)

    def process_files(self, file_paths):
        if not file_paths: return 0
        Path(config.DOCS_DIR).mkdir(parents=True, exist_ok=True)
        count = 0
        for fp in file_paths:
            try:
                path = Path(fp)
                if path.suffix.lower() == ".pdf":
                    md_text = pymupdf4llm.to_markdown(pymupdf.open(path))
                else:
                    md_text = path.read_text(encoding="utf-8", errors="ignore")
                
                parent_chunks = RecursiveCharacterTextSplitter(chunk_size=config.PARENT_CHUNK_SIZE).split_text(md_text)
                
                for i, p_text in enumerate(parent_chunks):
                    pid = f"{path.stem}_p{i}"
                    self.db.save_parent(pid, p_text, {"source": path.name})
                    
                    child_docs = self.splitter.create_documents([p_text])
                    for doc in child_docs:
                        doc.metadata.update({"parent_id": pid, "source": path.name})
                    
                    self.vector_store.add_documents(child_docs)
                count += 1
            except Exception as e:
                print(f"Failed to process {fp}: {e}")
        return count

class RAGAgent:
    def __init__(self, db: Database):
        self.db = db
        self.llm = ChatOllama(model=config.LLM_MODEL, temperature=0)
        self.vector_store = db.get_vector_store()
        self.graph = self._build_graph()
        self.thread_id = str(uuid.uuid4())

    def _build_graph(self):
        @tool
        def search_docs(query: str):
            """Search for relevant document chunks in the knowledge base."""
            try:
                docs = self.vector_store.similarity_search(query, k=4)
                if not docs:
                    return "No relevant documents found."
                return "\n\n".join([f"[ID:{d.metadata.get('parent_id')}] {d.page_content}" for d in docs])
            except Exception as e:
                return f"Search error: {e}"

        @tool
        def get_full_context(parent_id: str):
            """Retrieve the full parent document content using a parent ID found in search results."""
            return self.db.load_parent(parent_id)

        tools = [search_docs, get_full_context]
        llm_with_tools = self.llm.bind_tools(tools)

        sys_msg = SystemMessage(content="""You are a helpful assistant.
1. First, ALWAYS use 'search_docs' to look for information.
2. If a search result is cut off or you need more details, use 'get_full_context' with the specific ID.
3. Answer the user's question based ONLY on the retrieved information.
4. If you cannot find the answer, explicitly state that you don't know.""")

        def agent_node(state: MessagesState):
            messages = state["messages"]
            if not isinstance(messages[0], SystemMessage):
                messages = [sys_msg] + messages
            
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(tools))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")
        
        return workflow.compile(checkpointer=InMemorySaver())

    def chat(self, message: str):
        config_dict = {"configurable": {"thread_id": self.thread_id}}
        inputs = {"messages": [HumanMessage(content=message)]}
        
        try:
            result = self.graph.invoke(inputs, config_dict)
            
            last_msg = result["messages"][-1]
            
            if hasattr(last_msg, 'content') and last_msg.content:
                return last_msg.content
            
            return "I processed the data but couldn't generate a verbal response. Please try rephrasing."
            
        except Exception as e:
            return f"System Error: {str(e)}"

    def clear_history(self):
        self.thread_id = str(uuid.uuid4())