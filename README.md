# RAG Copilot

The Chainlit copilot for our platform [documentation](https://docs.getliteral.ai/introduction).

## How to use?

### The Chainlit Application

- `cp app/.env.example app/.env`
- Obtain a Literal API key [here](https://docs.getliteral.ai/python-client/get-started/authentication#how-to-get-my-api-key)
- Get an OpenAI API key [here](https://platform.openai.com/docs/quickstart/step-2-setup-your-api-key)
- Find your Pinecone API key [here](https://docs.pinecone.io/docs/authentication#finding-your-pinecone-api-key)
- `pip install -r app/requirements.txt`
- `chainlit run app/app.py`

### The Chainlit Copilot

- Make sure the Chainlit application is running.
- `python -m http.server 3004 --directory copilot`

### Embed the documentation
