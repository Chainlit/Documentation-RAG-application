import os

import chainlit as cl
import chromadb

from dotenv import load_dotenv
from openai import AsyncOpenAI
from literalai import AsyncLiteralClient

load_dotenv()
client = AsyncLiteralClient()
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path=f"{os.getcwd()}/chromadb")

cl.instrument_openai()


def create_chroma_collection(name):
    if name not in [collection.name for collection in chroma_client.list_collections()]:
        chroma_client.create_collection(name)
    return chroma_client.get_collection(name)


chroma_collection = create_chroma_collection("literal-doc-collection")


@cl.step(name="Embed", type="embedding")
async def embed(query, model="text-embedding-ada-002"):
    embedding = await openai_client.embeddings.create(input=query, model=model)

    return embedding.data[0].embedding


@cl.step(name="Retrieve", type="retrieval")
async def retrieve(embedding):
    if chroma_collection == None:
        raise Exception("Chroma collection not initialized")
    # Chroma returns documents and metadatas by default in an extension of TypedDict.
    return chroma_collection.query(query_embeddings=[embedding], n_results=5)


@cl.step(name="LLM", type="llm")
async def llm(
    prompt,
    chat_model="gpt-4-turbo-preview",
):
    messages = cl.user_session.get("messages", [])
    messages.append(prompt)
    settings = {"temperature": 0, "stream": True, "model": chat_model}
    stream = await openai_client.chat.completions.create(messages=messages, **settings)
    message = cl.message.Message(content="")
    await message.send()

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await message.stream_token(token)

    await message.update()
    messages.append({"role": "assistant", "content": message.content})
    cl.user_session.set("messages", messages)
    return message.content


@cl.step(name="Query", type="run")
async def run(query):
    embedding = await embed(query)
    query_result = await retrieve(embedding)
    contexts = []

    prompt = await client.api.get_prompt(name="RAG prompt")

    if not prompt:
        raise Exception("Prompt not found")

    # Selecting item 0 because we can query Chroma with several embeddings.
    for metadata in query_result["metadatas"][0]:
        contexts.append(metadata["text"])

    completion = await llm(prompt.format({"context": contexts, "question": query})[-1])

    return completion


@cl.on_chat_start
async def on_chat_start():
    prompt = await client.api.get_prompt(name="RAG prompt")

    if not prompt:
        raise Exception("Prompt not found")
    cl.user_session.set(
        "messages",
        [prompt.format()[0]],
    )


@cl.on_message
async def main(message: cl.Message):
    await run(message.content)
