from flask import Flask, request, render_template
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import weaviate
from langchain_community.vectorstores.weaviate import Weaviate
from langchain_community.embeddings.ollama import OllamaEmbeddings
#from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableLambda
from langchain_community.retrievers import WeaviateHybridSearchRetriever
import weaviate.classes as wvc
import os
import requests
import json

app = Flask(__name__)

#i need to get directory loading working
loader = TextLoader("/home/jonot480/Documents/paulgraham.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100,chunk_overlap=0)
docs = text_splitter.split_documents(documents)

client = weaviate.Client("http://localhost:8080")
#client = weaviate.connect_to_local()#spin up the docker container
questions = client.collections.create(
        name="CollectionTest",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(),  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
        generative_config=wvc.config.Configure.Generative.mistral()  # Ensure the `generative-openai` module is used for generative queries
    )

vectorstore=Weaviate(client,"CollectionTest","content")

#need to get collection queries without re-embedding everytime
weav = Weaviate.from_documents(
    docs, 
    embedding=OllamaEmbeddings(model='nomic-embed-text'),
    client=client
    #prefer_grpc=True, 
    #collection_name="my_documents",
)



retriever = weav.as_retriever()
#ensemble is an alt. retriever 

#llm = Ollama(model="jonorca3") #not sure why this one acts different
llm = ChatOllama(model="jonphi3")

after_rag_template="""
    <context>
    {context}
    </context>
    Question: {question}
    """
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    search_type = request.form['search_type']
    
    if search_type == 'keyword':
        # Keyword search
        found_docs = weav.similarity_search_by_text(query)
        after_rag_chain=(
            {"context":retriever,"question":RunnablePassthrough()}
            | after_rag_prompt
            | llm
            | StrOutputParser()    
        )
        results = after_rag_chain.invoke(query)
    elif search_type == 'vector':
        # Vector search
        found_docs = weav.similarity_search_by_vector(query)
        after_rag_chain=(
            {"context":RunnableLambda(lambda x: found_docs),"question":RunnablePassthrough()}
            | after_rag_prompt
            | llm
            | StrOutputParser()    
        )
        results = after_rag_chain.invoke(query)
    elif search_type == 'summarization':
        # Summarization search        
        found_docs = weav.similarity_search(query)
        chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
        results = chain.run(found_docs)
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run()