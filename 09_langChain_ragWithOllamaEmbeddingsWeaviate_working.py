from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community import embeddings
import qdrant_client
#from qdrant_client import QdrantClient
#import qdrant_client
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_community.chat_models import ChatOllama
from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import weaviate
import uuid
import chromadb
from chromadb.config import Settings
#from lan .utils import Callable
from langchain_core.runnables import RunnableLambda

model_local = ChatOllama(model="jonorca3")

loader = TextLoader("/home/jonot480/Documents/iamjon.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=10,chunk_overlap=5)
docs = text_splitter.split_documents(documents)

########################chroma
# client = chromadb.HttpClient(settings=Settings(allow_reset=True))
# client.reset()  # resets the database
# collection = client.create_collection("rag_chroma2")
# for doc in docs:
#     collection.add(
#         ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
#     )

# vectorstore = Chroma.from_documents(
#     documents = docs,
#     collection_name="rag_chroma2",
#     embedding= embeddings.OllamaEmbeddings(model='all-minilm')
# )

# retriever = vectorstore.as_retriever()
# doc_search = retriever.get_relevant_documents("who likes to pray")
######################end chroma

####################qdrant

#client = qdrant_client.QdrantClient(
#    "http://host.docker.internal:6333"
#)

# url = "http://localhost:6333"
# qdrant = Qdrant.from_documents(
#     docs, 
#     embeddings.OllamaEmbeddings(model='all-minilm'),
#     url=url, 
#     #prefer_grpc=True, 
#     collection_name="my_documents",
# )
client = weaviate.Client("http://localhost:8080")
weave = Weaviate.from_documents(
    docs, 
    embedding=OllamaEmbeddings(model='all-minilm'),
    client=client
    #prefer_grpc=True, 
    #collection_name="my_documents",
)

query = "what is the effective date?"
found_docs = weave.similarity_search(query)

retriever2 = weave.as_retriever()
####################

after_rag_template="""
<context>
{context}
</context>
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain=(
    {"context":RunnableLambda(lambda x: found_docs),"question":RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()    
)

print(after_rag_chain.invoke("what is the effective date?"))
