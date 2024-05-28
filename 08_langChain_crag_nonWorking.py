from langchain_community.llms import Ollama
#from langchain_core.prompts import TextPrompt
from langchain_community.embeddings import OllamaEmbeddings
#from langchain.storage import VectorDatabaseClient
from langchain_community.vectorstores.qdrant import Qdrant
#from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient, models

print()


ollama_emb = OllamaEmbeddings(
    model="embedding-mistral",
)
r1 = ollama_emb.embed_documents(
     [
         "Alpha is the first letter of Greek alphabet",
         "Beta is the second letter of Greek alphabet",
     ]
)
r2 = ollama_emb.embed_query(
     "What is the second letter of Greek alphabet"
)

doc_store = Qdrant.from_texts(
       [{ "id": 1, "format": "PDF", "text": "RIN = 365A-BB" }]
    , ollama_emb, url="http://localhost:6333", collection_name="test2"
)

print ("done")

#client = QdrantClient(url="http://localhost:6333")

'''client.upsert(
    collection_name="test2",
    points=[
        models.PointStruct(
            id=1,
            vector=[0.05, 0.61, 0.76, 0.74],
            payload={
                "city": "Berlin",
                "price": 1.99,
            },
        ),
        models.PointStruct(
            id=2,
            vector=[0.19, 0.81, 0.75, 0.11],
            payload={
                "city": ["Berlin", "London"],
                "price": 1.99,
            },
        ),
        models.PointStruct(
            id=3,
            vector=[0.36, 0.55, 0.47, 0.94],
            payload={
                "city": ["Berlin", "Moscow"],
                "price": [1.99, 2.99],
            },
        ),
    ],
)'''




# response2 = client.scroll(
#     collection_name="test2",
#     scroll_filter=models.Filter(
#         must=[
#             models.FieldCondition(
#                 key="city",
#                 match=models.MatchValue(value="London"),
#             ),
#             models.FieldCondition(
#                 key="color",
#                 match=models.MatchValue(value="green"),
#             ),
#         ]
#     ),
# )

# print(response2)




# Connect to Qdrant vector database
#client = Qdrant(host="localhost", port=6333)

# Define the Ollama model
#model_name = "orca-mini"

# Load the sentence transformers model for embedding
#embedding_model = SentenceTransformerEmbeddings("embedding-mistral")

#llm = Ollama(model_name=model_name)  
#print(llm.invoke("hi"))

