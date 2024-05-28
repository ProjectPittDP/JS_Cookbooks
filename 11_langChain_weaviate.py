from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from weaviate import WeaviateClient
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_community.embeddings.ollama import OllamaEmbeddings
import weaviate
from weaviate.collections.classes import config_vectorizers

weaviate_client = weaviate.connect_to_local("localhost","8081")#v4
collections = weaviate_client.collections.list_all()
collTest = weaviate_client.collections.get("Af29061962debf95ac6117d5dd3604cf8327394cacc5d43ddb3e5c95f14343f")

response = collTest.query.fetch_objects()

for o in response.objects:
   print(o.properties["text"])




loader = TextLoader("/home/jonot480/Documents/paulgraham.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100,chunk_overlap=0)
docs = text_splitter.split_documents(documents)

#weaviate_client = weaviate.connect_to_local("http://localhost:8081")#v4

client = weaviate.Client("http://localhost:8081")

#vectorstore=Weaviate(client,"CollectionTest","content")

#weaviate = Weaviate(client, index_name, text_key)

weav = Weaviate.from_documents(
    docs, 
    embedding=OllamaEmbeddings(model='nomic-embed-text'),
    client=client
    #prefer_grpc=True, 
    #collection_name="my_documents",
)

found_docs = weav.similarity_search("what is the effective date",k=4)

print(found_docs)