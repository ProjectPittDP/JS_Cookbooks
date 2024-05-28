from langchain_community.document_loaders import TextLoader
# from langchain_chroma import Chroma
# from langchain_community import embeddings
#import qdrant_client
from qdrant_client import QdrantClient
import qdrant_client
# from langchain_community.vectorstores.qdrant import Qdrant
# from langchain_community.chat_models import ChatOllama
# from langchain_community.llms.ollama import Ollama
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import uuid
import chromadb
from chromadb.config import Settings

loader = TextLoader("/home/jonot480/Documents/iamjon.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=10,chunk_overlap=5)
docs = text_splitter.split_documents(documents)

#from weaviate import WeaviateClient
#from langchain_community.vectorstores.weaviate import Weaviate

########################chroma
# client = chromadb.HttpClient(settings=Settings(allow_reset=True))
# client.reset()  # resets the database
# collection = client.create_collection("rag_chroma2")
# for doc in docs:
#     collection.add(
#         ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
#     )

#look into the vector db admin tool 
# vectorstore = Chroma.from_documents(
#     documents = docs,
#     collection_name="rag_chroma2",
#     embedding= embeddings.OllamaEmbeddings(model='all-minilm')
# )

#retriever = vectorstore.as_retriever()
######################end chroma

####################qdrant

client = QdrantClient(url="http://localhost:6333")


# client = qdrant_client.QdrantClient(
#     "http://host.docker.internal:6333"
# )

# url = "http://localhost:6333"
# qdrant = Qdrant.from_documents(
#      docs, 
#      embeddings.OllamaEmbeddings(model='all-minilm'),
#      url=url, 
#      #prefer_grpc=True, 
#      collection_name="my_documents",
#  )

query = "vertigo"
found_docs = client.query(collection_name="my_documents",query_text=query)
print(found_docs)
#retriever2 = qdrant.as_retriever()
####################

