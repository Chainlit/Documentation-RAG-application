import chainlit as cl
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

pinecone_client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_spec = ServerlessSpec(cloud="aws", region="us-west-2")


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
    sys_prompt="You are a helpful assistant that always answers questions. Keep it short, and prefer responding with code.",
    chat_model="gpt-3.5-turbo",
):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    settings = {"temperature": 0, "stream": True, "model": chat_model}
    stream = await openai_client.chat.completions.create(messages=messages, **settings)
    message = cl.message.Message(content="")
    await message.send()

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await message.stream_token(token)
    current_step = cl.context.current_step

    if current_step:
        current_step.generation = cl.ChatGeneration(
            provider="openai-chat",
            completion=message.content,
            messages=[
                cl.GenerationMessage(formatted=m["content"], role=m["role"])
                for m in messages
            ],
            settings=settings,
        )
    await message.update()
    return message.content


@cl.step(name="Query", type="run")
async def run(query):
    embedding = await embed(query)
    stored_embeddings = await retrieve(embedding)
    contexts = []

    for match in stored_embeddings["matches"]:
        contexts.append(match["metadata"]["text"])
    prompt = (
        "Answer the question based on the context below.\n\n"
        + "Context:\n"
        + "\n".join(contexts)
        + f"\n\nQuestion: {query}\nAnswer:"
    )
    completion = await llm(prompt)

    return completion


@cl.on_message
async def main(message: cl.Message):
    await run(message.content)
