from langchain_ollama import ChatOllama
from config import OLLAMA_API_KEY

llm = ChatOllama(
    model="gpt-oss:120b",
    base_url="https://ollama.com",
    client_kwargs={"headers": {"Authorization": "Bearer " + OLLAMA_API_KEY}, "timeout": 120},
)
print(llm.invoke("Say hi in one short sentence").content)