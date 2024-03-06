import chainlit as cl
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec
from literalai import LiteralClient

load_dotenv()

client = LiteralClient()
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

pinecone_client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_spec = ServerlessSpec(
    cloud=os.environ.get("PINECONE_CLOUD"), region=os.environ.get("PINECONE_REGION")
)

cl.instrument_openai()


def create_pinecone_index(name, client, spec):
    if name not in client.list_indexes().names():
        client.create_index(name, dimension=1536, metric="cosine", spec=spec)
    return pinecone_client.Index(name)


pinecone_index = create_pinecone_index(
    "literal-rag-index", pinecone_client, pinecone_spec
)


@cl.step(name="Embed", type="embedding")
async def embed(query, model="text-embedding-ada-002"):
    embedding = await openai_client.embeddings.create(input=query, model=model)

    return embedding.data[0].embedding


@cl.step(name="Retrieve", type="retrieval")
async def retrieve(embedding):
    if pinecone_index == None:
        raise Exception("Pinecone index not initialized")
    response = pinecone_index.query(vector=embedding, top_k=5, include_metadata=True)

    return response


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
    stored_embeddings = await retrieve(embedding)
    contexts = []
    prompt = await client.api.get_prompt(name="RAG prompt")

    if not prompt:
        raise Exception("Prompt not found")
    for match in stored_embeddings["matches"]:
        contexts.append(match["metadata"]["text"])

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
