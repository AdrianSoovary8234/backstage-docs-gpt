
import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

index_name = "backstage-docs-embeddings"

def delete_index(name: str):

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    try: 
        pc.delete_index(name)
        print(f"Deleted index {name}")
    except Exception as e:
        print(f"Error deleting index: {e}")


delete_index(index_name)
