import sqlite3
import openai
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

# Initialize your vector database (this is hypothetical and will depend on your specific vector db setup)
chroma = Chroma()

# Connect to SQLite database
conn = sqlite3.connect('./Data/Chinook.db')
cursor = conn.cursor()

# Fetch data from SQLite database
cursor.execute("SELECT * FROM Employee")
rows = cursor.fetchall()
#vectorstore = Chroma.from_documents(
#    documents=all_splits, embedding=OpenAIEmbeddings()
#)

for row in rows:
    # Assuming each row has text data in the second column
    text_data = row[1]
        
    embedding=OpenAIEmbeddings()
    #chroma.insert(embedding, text_data)

    #response = openai.Embedding.create(
    #   input=[text_data],
    #   engine="text-similarity-davinci-001"  # or another embeddings model
    #)
    #embedding = response['data'][0]['embedding']
    
    # Store the embeddings in Chroma
    # Replace 'add_vector' with the actual method to add data to Chroma
    # You might need to provide an ID or key for each piece of data
    #chroma. .add_vector(key=row[0], vector=embedding, metadata={'text_data': text_data})


# Now, assuming you want to use this data to prompt GPT-3.5

# Define your prompt based on the data you have in your vector db
# For demonstration, we'll just fetch one record's text data
record = chroma.fetch_one()
prompt = record['text_data']  # Hypothetical field containing the text data

# Setup OpenAI API (make sure to have your API key)
#openai.api_key = 'your-api-key'

# Call the OpenAI API with the prompt
response = openai.Completion.create(
  engine="text-davinci-003",  # or another GPT-3.5 model variant
  prompt=prompt,
  max_tokens=100
)

# Print the GPT-3.5 generated response
print(response.choices[0].text.strip())

# Close the database connection
conn.close()
