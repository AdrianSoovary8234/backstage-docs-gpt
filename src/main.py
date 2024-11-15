import time
import requests
import os
import re
import uuid
from bs4 import BeautifulSoup
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, Index, ServerlessSpec


# Load environment variables from .env file
load_dotenv()

class EmbeddingResult:
    embedding: List[float]  
    chunk: str

def fetch_sitemap_urls(sitemap_url: str, exclude_patterns: List[str] = None) -> List[str]:
    """
    Fetches URLs from the sitemap and excludes any that match the specified patterns.
    
    Parameters:
    - sitemap_url: The URL to the sitemap.xml file.
    - exclude_patterns: List of regex patterns to exclude from the URLs.

    Returns:
    - A list of URLs extracted from the sitemap.
    """
    response = requests.get(sitemap_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "xml")
    
    urls = [url_loc.text for url_loc in soup.find_all("loc")]
    
    if exclude_patterns:
        urls = [
            url for url in urls
            if not any(re.search(pattern, url) for pattern in exclude_patterns)
        ]
    
    return urls

def extract_main_content(url: str) -> str:
    """
    Fetches the URL and extracts content within the <main> tag.
    
    Parameters:
    - url: The URL to fetch and parse.

    Returns:
    - A string containing the content of the <main> tag, if found. Otherwise, an empty string.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        page_soup = BeautifulSoup(response.content, "lxml")
        main_content = page_soup.find("main").prettify()

        return  get_semantic_content(main_content) if main_content else ""
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""
    

def get_semantic_content (htmlString: str) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role': 'user', 'content': get_prompt(htmlString)}],
        temperature=0.1
    )
    print('successful text extraction')
    return response.choices[0].message.content


def get_prompt (htmlString: str) -> str:
    return f"""Extract the main text content from the following html. 
        There shouldn't be any html tags in the result. Also, it should be structured 
        in a way that meaningful paragraphs stay together. If there are code examples 
        in the HTML, they should be maintained in a block and not contain multiple line breaks. 
        The returned string will be handled by a recursive text splitter with the following 
        selector setting ["\\n\\n", ". ", " "]
        This is the HTML: {htmlString}"""


def handle_urls(urls: List[str]) -> None:
    index_name = "backstage-docs-embeddings"

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    get_or_create_index(pc, index_name)

    index = pc.Index(index_name)

    for url in urls:
        print(f"Processing URL: {url}")
        main_content = extract_main_content(url)
        if main_content:
            main_content_chunks = split_recursively(main_content)
            embedded_chunks = embed_chunks(main_content_chunks)
            store_embedding(embedded_chunks, index)
        else:
            print(f"No <main> content found for {url}.")

def split_recursively (text: str) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=20,
        separators=["\n\n", ". ", " "]
    )
    return text_splitter.split_text(text)


def embed_chunks (chunks: List[str]) -> List[EmbeddingResult]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = []
    for chunk in chunks:
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            print(f"Embedded chunk: {chunk[:50]}...")
            embeddings.append({'embedding': response.data[0].embedding, 'chunk': chunk})
        except Exception as e:
            print(f"Error embedding chunk: {e}")

    return embeddings

def store_embedding (embeddingsWithText: List[EmbeddingResult], index: Index) -> List[str]:
    upsert_embeddings(embeddingsWithText, index)


def upsert_embeddings (embeddingsWithText: List[EmbeddingResult], index: Index) -> None:
    for i, embeddingWithText in enumerate(embeddingsWithText):
        print(f"Upserting chunk {i} with embedding {embeddingWithText['chunk'][:50]}...")
        try: 
            embeddingVector = embeddingWithText['embedding']
            chunk = embeddingWithText['chunk']
            
            id = f"chunk{uuid.uuid4()}-{i}"
            index.upsert(vectors=[{"id": id, "values": embeddingVector, 'metadata': {'textChunk': chunk}}])
            print(f"Upserted chunk {i} {embeddingWithText['chunk'][:50]}... with embedding {embeddingWithText['embedding'][:50]}...")
            
        except Exception as e:
            print(f"Error upserting chunk: {e}")
    

    


def get_or_create_index (pc: Pinecone, index_name: str) -> None:
    if not pc.has_index(index_name):
        print(f"Creating index {index_name}")
        pc.create_index(
            index_name, 
            dimension=1536, 
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            )
        )
        print('Index created')
        time.sleep(2)

def main(sitemap_url: str, exclude_patterns: List[str] = None):
    """
    Main function to extract URLs from sitemap, filter them, and extract <main> content from each.
    
    Parameters:
    - sitemap_url: URL to the sitemap XML.
    - exclude_patterns: List of regex patterns to exclude certain URLs.
    """
    urls = fetch_sitemap_urls(sitemap_url, exclude_patterns)
    print(urls.__len__())

    handle_urls(urls)

# Example usage
sitemap_url = "https://backstage.io/sitemap"
exclude_patterns = [r"/docs/next/", r"/docs/releases/", r"/docs/reference/"]
main(sitemap_url, exclude_patterns)