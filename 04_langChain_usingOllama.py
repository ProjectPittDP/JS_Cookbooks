from langchain_community.llms import Ollama
from langchain_community.chat_models.ollama import ChatOllama
import os
from langsmith import traceable

os.environ["LANGCHAIN_TRACING_V2"] = "true"

#llm = Ollama(model="orca-mini")
llm = ChatOllama(model="phi3")
print(llm.invoke("bye"))


