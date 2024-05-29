# RAG Starter
This is a Starter Repository for RAG. In this we will be building out a PDF RAG Agent to chat with. The documents in pdf format will be the contect corpus to use for embeddings that we will index in the pinecone graph index.

## Initial Setup
- fork and clone this repository
- Create a pinecone account [here](https://www.pinecone.io/)
- setup an index and call it `pineidx`
- create a Virtual Env `python3 -m venv .venv`
- install the requirements `pip install -r requirements.txt`
- look over the python files and read the comments
- get a pinecone key from your account
- export the openaikey `export OPENAI_API_KEY=pk_your_key_here` 
- export the pinecone key `export PINECONE_API_KEY=d0_your_pinecone_key_here`
- get started writing out the upload logic
- add some pdf documents in to the docs directory
- run the upload.py `python upload.py`
- if it was successful then we can move on to building out the main file logic
- once we have the main file done try it out `python main.py`
- if it is working we can build a repl and make the prompt be more useful maybe even choose more pdf files to upload
