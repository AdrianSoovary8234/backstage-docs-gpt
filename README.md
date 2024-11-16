# Backstage Docs GPT

This is an attempt to use an RAG GPT to be helpful for Backstage devs as the amount of content and resources in the web are quite limited
It contains of 3 different subrepos

# scrape-docs

Python script to scrape the backstage docs, extract readable text and chunk it into semantic pieces of text.
Create embeddings for the text and store it into Pinecone vector database.

# frontend

Used chatbotui to use a readily made solution for frontend handling. Needs some adjustments to access our backend instead of OAI or the like. We might be able to just adjust it and use api routes altogether instead of a separate python backen.

# backend

lightweight python flask application offering an endpoint that takes a search query parameter, embeds it, queries the database for similar text chunks, and enriches the query to GPT-4o-mini with that content.
