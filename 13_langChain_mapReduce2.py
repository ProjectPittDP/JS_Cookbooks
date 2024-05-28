from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os 
from langchain.chains.question_answering import load_qa_chain
import weaviate
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_community.embeddings.ollama import OllamaEmbeddings


loader = TextLoader("/home/jonot480/Documents/iamjon.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100,chunk_overlap=0)
docs = text_splitter.split_documents(documents)

weaviate_client = weaviate.connect_to_local()#spin up the docker container

client = weaviate.Client("http://localhost:8080")

#jeopardy = weaviate_client.collections.get("CollectionTest")
vectorstore=Weaviate(client,"CollectionTest","content") 
#weaviate_client.collections.delete("CollectionTest")  

#weaviate = Weaviate(client, index_name, text_key)
#load my document into the vector
weav = Weaviate.from_documents(
    docs, 
    embedding=OllamaEmbeddings(model='all-minilm'),
    client=client
    #prefer_grpc=True, 
    #collection_name="my_documents",
)

found_docs = weav.similarity_search("what is the effective date?",k=4)
#i think this is a hybrid search of keyboard and vector?

from langchain_community.llms import Ollama
llm = Ollama(model="orca-mini")
#llm = OpenAI(openai_api_key=OPENAI_API_KEY)

#stuff isn't going to work with a 600 page doc 
chain = load_summarize_chain(llm, chain_type="stuff", verbose=True)
chain.run(found_docs)
#load_map_reduce_chain
chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
#res = chain.run(found_docs)
chain.run(found_docs)

chain = load_summarize_chain(llm, chain_type="refine", verbose=True)
#res2 = chain.run(found_docs)#i don't think you can pass res here, but maybe?
chain.run(found_docs)

chain = load_qa_chain(llm, chain_type="map_rerank", verbose=True, return_intermediate_steps=True)
query = "what can fierceness turn into?"
print(chain({"input_documents": found_docs, "question": query})) #this isn't working need 