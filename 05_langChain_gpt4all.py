from langchain_community.llms import GPT4All

llm = GPT4All(
    model="/home/jonoupstairswork/.local/share/nomic.ai/GPT4All/gpt4all-falcon-newbpe-q4_0.gguf"
)
response = llm.invoke("The first man on the moon was who?")
print(response)


