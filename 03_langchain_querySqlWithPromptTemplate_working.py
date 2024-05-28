import sqlite3
from langchain_core.prompts import ChatPromptTemplate

template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)


from langchain_community.utilities import SQLDatabase

#conn = sqlite3.connect('Chinook.db')
conn = SQLDatabase.from_uri("sqlite:///DB/Chinook.db")
#cursor = conn.cursor()
#cursor.execute("SELECT * FROM Employee")
#rows = cursor.fetchall()
#db = SQLDatabase.from_uri("sqlite:///./Chinook.db") #old not working

def get_schema(_):
    return conn.get_table_info()

def run_query(query):
    return conn.run(query)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

sql_response.invoke({"question": "How many employees are there in the employee table?"})

template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_template(template)

full_chain = (
    RunnablePassthrough.assign(query=sql_response).assign(
        schema=get_schema,
        response=lambda x: conn.run(x["query"]),
    )
    | prompt_response
    | model
)

rsp = full_chain.invoke({"question": "How many employees are there?"})
print(rsp)