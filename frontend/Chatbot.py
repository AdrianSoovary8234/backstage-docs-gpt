from openai import OpenAI
import streamlit as st
from requests import post, Response
import json


def create_generator_from_response(response: Response) -> None:
    assistant_response = ''
    for chunk in response.iter_content(chunk_size=1024):  # Read in 1KB chunks
        if chunk:  # Skip empty chunks
            try:
                chunk_decoded = chunk.decode('utf-8').strip()
                if not chunk_decoded:
                    continue

                chunk_data = json.loads(chunk_decoded)

                if 'choices' in chunk_data and 'finish_reason' in chunk_data['choices'][0]:
                    if chunk_data['choices'][0]['finish_reason'] == 'stop':
                        continue

                # Yield the parsed JSON
                content =  chunk_data['choices'][0]['delta']['content']
                assistant_response += content

                yield content
            except json.JSONDecodeError as e:
                # If JSON decoding fails, print an error and continue
                print(f"JSON decoding error: {e}. Raw chunk: {chunk.decode('utf-8')}")
                continue
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

st.title("💬 Chat with the backstage.io")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)


    response = post('http://127.0.0.1:5000/ask', json={'query': prompt}, headers={'Content-Type': 'application/json'}, stream=True)

    with st.chat_message("assistant"):
        st.write_stream(create_generator_from_response(response))
    
