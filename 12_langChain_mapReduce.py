from langchain.chains import (
    StuffDocumentsChain,
    LLMChain,
    ReduceDocumentsChain,
    MapReduceDocumentsChain,
)
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# This controls how each document will be formatted. Specifically,
# it will be passed to `format_document` - see that function for more
# details.


loader = TextLoader("/home/jonot480/Documents/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=100,chunk_overlap=0)

docs = text_splitter.split_documents(documents)
#docs =  text_splitter.split_text(documents)

document_prompt = PromptTemplate(
    input_variables=["page_content"],
     template="{page_content}"
)
document_variable_name = "context"
llm = Ollama(model="orca-mini")
# The prompt here should take as an input variable the
# `document_variable_name`
prompt = PromptTemplate.from_template(
    "Summarize this content: {context}"
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
# We now define how to combine these summaries
reduce_prompt = PromptTemplate.from_template(
    "Combine these summaries: {context}"
)
reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_prompt)
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name
)
reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
)
chain = MapReduceDocumentsChain(
    llm_chain=llm_chain,
    reduce_documents_chain=reduce_documents_chain,
)
# If we wanted to, we could also pass in collapse_documents_chain
# which is specifically aimed at collapsing documents BEFORE
# the final call.
prompt = PromptTemplate.from_template(
    "Collapse this content: {context}"
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
collapse_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name
)
reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    collapse_documents_chain=collapse_documents_chain,
)
chain = MapReduceDocumentsChain(
    llm_chain=llm_chain,
    reduce_documents_chain=reduce_documents_chain,
)






# from langchain.chains import AnalyzeDocumentChain
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.chat_models import ChatOllama
# from langchain_community.vectorstores.weaviate import Weaviate
# from langchain_community.embeddings.ollama import OllamaEmbeddings
# import weaviate
# from langchain.prompts import PromptTemplate

# model_local = ChatOllama(model="jonorca3")

# loader = TextLoader("/home/jonot480/Documents/state_of_the_union.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=100,chunk_overlap=0)
# #docs = text_splitter.split_documents(documents)
# docs =  text_splitter.split_text(documents)

# client = weaviate.Client("http://localhost:8080")
# vectorstore=Weaviate(client,"CollectionTest","content")

# #weaviate = Weaviate(client, index_name, text_key)

# weav = Weaviate.from_texts(
#     docs, 
#     embedding=OllamaEmbeddings(model='all-minilm'),
#     client=client
#     #prefer_grpc=True, 
#     #collection_name="my_documents",
# )
# from langchain.chains.summarize import load_summarize_chain

# prompt_template = """Write a concise summary of the following:


# {text}


# CONCISE SUMMARY IN INDONESIAN:"""
# PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

# summary_chain = load_summarize_chain(model, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT)


# after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
# after_rag_chain=(
#     {"context":retriever2,"question":RunnablePassthrough()}
#     | after_rag_prompt
#     | model_local
#     | StrOutputParser()    
# )
# print(after_rag_chain.invoke("what is the effective date?"))


# summarize_document_chain = AnalyzeDocumentChain(
#     combine_docs_chain=chain, text_splitter=text_splitter
# )
# summarize_document_chain.run(docs[0].page_content)


