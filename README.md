# Documentation - RAG application

The RAG chatbot application for the Literal platform [documentation](https://docs.getliteral.ai/introduction).

## How to use?

### Embed the documentation

To upload the latest Literal documentation embeddings to Pinecone:

- `cp embed-documentation/.env.example embed-documentation/.env`
- Get an OpenAI API key [here](https://platform.openai.com/docs/quickstart/step-2-setup-your-api-key)
- Find your Pinecone API key [here](https://docs.pinecone.io/docs/authentication#finding-your-pinecone-api-key)
- `pip install -r embed-documentation/requirements.txt`
- `./embed-documentation/main.sh`

### The Chainlit Application

- `cp app/.env.example app/.env`
- Obtain a Literal API key [here](https://docs.getliteral.ai/python-client/get-started/authentication#how-to-get-my-api-key)
- Get an OpenAI API key [here](https://platform.openai.com/docs/quickstart/step-2-setup-your-api-key)
- Find your Pinecone API key [here](https://docs.pinecone.io/docs/authentication#finding-your-pinecone-api-key)
- `pip install -r app/requirements.txt`
- `chainlit run app/app.py`

