from typing import List
from flask import Flask, Response, request
from openai import OpenAI
import time
import os
from dotenv import load_dotenv 
from pinecone import Pinecone, Index,  ScoredVector

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

pc_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pc_key)


def get_pc_index() -> Index: 
    return pc.Index("backstage-docs-embeddings")

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        dimensions=1536
    )
    return response.data[0].embedding
    
        
def get_most_relevant_context(query_vector: str) -> str:
    index = get_pc_index()

    return index.query(vector=query_vector, top_k=5, include_metadata=True)

def get_raw_text_from_pinecone_responses(content: List[ScoredVector]) -> List[str]:
    return [c.metadata['textChunk'] for c in content]

def build_system_prompt(vector_responses: List[ScoredVector]) -> dict:
    
    context_text_fragments = get_raw_text_from_pinecone_responses(vector_responses)
    context_text = "\n".join(context_text_fragments)

    return {
      "role": "system",
      "content": f"""
          You are a helpful assistant that helps developers with their questions. Here is a conversation with a developer who is asking about Spotify's open source tool Backstage.
          In the following you will find several text chunks that have been found relevant to the user's question:
          ---- START CONTEXT ----
          {context_text}
          ---- END CONTEXT ----
      """
}

def build_user_prompt(query: str) -> dict:
    return {
      "role": "user",
      "content": f""" ---- START QUESTION ----
      {query}
      ---- END QUESTION ----"""
    }


    

    

    
    

def stream_openai_response(query_string: str):
    embedded_query = get_embedding(query_string)
    vector_responses = get_most_relevant_context(embedded_query)

    system_prompt = build_system_prompt( vector_responses.matches)
    user_prompt = build_user_prompt(query_string)
    

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[system_prompt, user_prompt],
            temperature=0.4,
            stream=True  
        )

        print("Starting OpenAI chat...")
        
        for chunk in response:
            if "choices" in chunk:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    yield f"data: {delta['content']}\n\n"

    except Exception as e:
        yield f"data: [Error] {str(e)}\n\n"


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data:
        return "No data provided", 400
    
    query_string = data.get("query")

    if not query_string:
        return "No query provided", 400

    return Response(stream_openai_response(query_string), content_type='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)

