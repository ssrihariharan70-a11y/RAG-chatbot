import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
PDF_DIRECTORY = "./documents"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150