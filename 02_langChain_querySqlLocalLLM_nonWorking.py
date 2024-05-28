#need to use a langChain integration wrapper with the LLM you want to use

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
#from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
#from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from langchain_core.prompts import PromptTemplate

db = SQLDatabase.from_uri("sqlite:///DB/Chinook.db")
#print(db.dialect)
#print(db.get_usable_table_names())
#db.run("SELECT * FROM Artist LIMIT 10;")
#print(db.)
#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = Ollama(model="orca-mini")
#llm = ChatMistralAI(model="mistral", request_timeout=300.0) #doesn't work 
chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "How many employees are there"})
print(response)
print(db.run(response))


