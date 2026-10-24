import os
import re
import time
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from config import *

SECTION_HEADINGS = [
    "Purpose and scope",
    "Ownership and responsibilities",
    "Standard workflow",
    "Controls and evidence",
    "Decision guidance",
    "Example scenario",
    "Service quality and review",
    "Common questions",
    "Related search terms",
]
_SECTION_SPLIT = re.compile(
    r"\n(?=(?:" + "|".join(re.escape(h) for h in SECTION_HEADINGS) + r")\n)"
)


def split_page_into_sections(text):
    """Split one module page into (header, [section, ...]).
    header = module title lines, so every chunk keeps its module context."""
    parts = _SECTION_SPLIT.split(text)
    if len(parts) < 3:
        return None, []
    header_lines = [
        l.strip()
        for l in parts[0].splitlines()
        if "Reference Module" in l or "Module ID" in l
    ]
    return " | ".join(header_lines), [p.strip() for p in parts[1:] if p.strip()]


class LocalEmbeddings(Embeddings):
    """Free local embeddings (chromadb's built-in ONNX model, no torch needed)"""

    def __init__(self):
        self.ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return [[float(x) for x in e] for e in self.ef(texts)]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class RAGChatbot:
    def __init__(self):
        self.documents = []
        self.vector_store = None
        self.prompt = None

    def load_documents(self):
        print("Loading documents...")
        try:
            loader = PyPDFDirectoryLoader(PDF_DIRECTORY, recursive=True)
            self.documents = loader.load()
            print(f"Loaded {len(self.documents)} pages")
            return len(self.documents)
        except Exception as e:
            print(f"Error loading documents: {e}")
            return 0

    def chunk_documents(self):
        print("Chunking documents (section-aware)...")
        fallback = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = []
        for doc in self.documents:
            header, sections = split_page_into_sections(doc.page_content)
            if not sections:  # page without known headings: normal splitting
                chunks.extend(fallback.split_documents([doc]))
                continue
            for sec in sections:
                text = (header + "\n" + sec) if header else sec
                if len(text) > 1200:  # very long section: split it further
                    pieces = fallback.split_text(sec)
                    for p in pieces:
                        chunks.append(
                            Document(
                                page_content=(header + "\n" + p) if header else p,
                                metadata=dict(doc.metadata),
                            )
                        )
                else:
                    chunks.append(
                        Document(page_content=text, metadata=dict(doc.metadata))
                    )
        print(f"Created {len(chunks)} chunks")
        return chunks

    def create_vector_store(self, chunks=None):
        if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
            print("Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=LocalEmbeddings(),
            )
        else:
            print("Creating embeddings...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=LocalEmbeddings(),
                persist_directory=CHROMA_DB_PATH,
            )
        print("Vector store ready")

    def _make_llm(self):
        # Fresh client every call, so a dropped/stale connection is never reused
        return ChatOllama(
            model="gpt-oss:120b",
            base_url="https://ollama.com",
            client_kwargs={
                "headers": {"Authorization": "Bearer " + OLLAMA_API_KEY},
                "timeout": 120,
            },
        )

    def build_qa_chain(self):
        print("Building QA chain...")
        self.prompt = ChatPromptTemplate.from_template(
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            "Context:\n{context}\n\nQuestion: {question}"
        )
        print("QA chain ready")

    def ask(self, question):
        docs = self.vector_store.similarity_search(question, k=10)
        context = "\n\n".join(d.page_content for d in docs)
        last_error = None
        for attempt in range(5):  # retry on flaky network
            try:
                response = (self.prompt | self._make_llm()).invoke(
                    {"context": context, "question": question}
                )
                return response.content, docs
            except Exception as e:
                last_error = e
                time.sleep(2 * (attempt + 1))
        raise last_error