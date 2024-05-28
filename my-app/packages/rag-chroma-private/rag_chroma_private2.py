from langserve.client import RemoteRunnable

rag_app = RemoteRunnable("http://0.0.0.0:8001/rag_chroma_private/")
rag_app.invoke("How does agent memory work?")