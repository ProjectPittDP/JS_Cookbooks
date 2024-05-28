from langchain_community.llms import Ollama
import langsmith

langsmith_client=langsmith.Client(
    api_key='ls_0a4028d0b8844641b9c9416a1b026eb7',
    api_url='https://api.smith.langchain.com'    
)


llm = Ollama(model="orca-mini")
print(llm.invoke("hi my name is jon.  what is my name?"))

