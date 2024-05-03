import os
import json
import asyncio
import chainlit as cl

from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec  # type: ignore

from literalai import LiteralClient

client = LiteralClient()
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

pinecone_client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_spec = ServerlessSpec(
    cloud=os.environ.get("PINECONE_CLOUD"),
    region=os.environ.get("PINECONE_REGION"),
)

cl.instrument_openai()

prompt_path = os.path.join(os.getcwd(), "app/prompts/rag.json")

# We load the RAG prompt in Literal to track prompt iteration and
# enable LLM replays from Literal AI.
with open(prompt_path, "r") as f:
    rag_prompt = json.load(f)

    prompt = client.api.get_or_create_prompt(
        name=rag_prompt["name"],
        template_messages=rag_prompt["template_messages"],
        settings=rag_prompt["settings"],
        tools=rag_prompt["tools"],
    )


def create_pinecone_index(name, client, spec):
    """
    Create a Pinecone index if it does not already exist.
    """
    if name not in client.list_indexes().names():
        client.create_index(name, dimension=1536, metric="cosine", spec=spec)
    return pinecone_client.Index(name)


pinecone_index = create_pinecone_index(
    "literal-rag-index", pinecone_client, pinecone_spec
)


@cl.step(name="Embed", type="embedding")
async def embed(question, model="text-embedding-ada-002"):
    """
    Embed a question using the specified model and return the embedding.
    """
    embedding = await openai_client.embeddings.create(input=question, model=model)

    return embedding.data[0].embedding


@cl.step(name="Retrieve", type="retrieval")
async def retrieve(embedding, top_k):
    """
    Retrieve top_k closest vectors from the Pinecone index using the provided embedding.
    """
    if pinecone_index == None:
        raise Exception("Pinecone index not initialized")
    response = pinecone_index.query(
        vector=embedding, top_k=top_k, include_metadata=True
    )
    return response.to_dict()


@cl.step(name="Retrieval", type="tool")
async def get_relevant_documentation_chunks(question, top_k=5):
    """
    Retrieve relevant documentation chunks based on the question embedding.
    """
    embedding = await embed(question)

    retrieved_chunks = await retrieve(embedding, top_k)

    return [match["metadata"]["text"] for match in retrieved_chunks["matches"]]


async def llm_tool(question):
    """
    Generate a response from the LLM based on the user's question.
    """
    messages = cl.user_session.get("messages", []) or []
    messages.append({"role": "user", "content": question})

    settings = cl.user_session.get("settings", {}) or {}

    response = await openai_client.chat.completions.create(
        messages=messages,
        **settings,
        tools=cl.user_session.get("tools"),
        tool_choice="auto"
    )

    response_message = response.choices[0].message
    messages.append(response_message)
    return response_message


async def run_multiple(tool_calls):
    """
    Execute multiple tool calls asynchronously.
    """
    available_tools = {
        "get_relevant_documentation_chunks": get_relevant_documentation_chunks
    }

    async def run_single(tool_call):
        function_name = tool_call.function.name
        function_to_call = available_tools[function_name]
        function_args = json.loads(tool_call.function.arguments)

        function_response = await function_to_call(
            question=function_args.get("question"),
            top_k=function_args.get("top_k"),
        )
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": "\n".join(function_response),
        }

    # Run tool calls in parallel.
    tool_results = await asyncio.gather(
        *(run_single(tool_call) for tool_call in tool_calls)
    )
    return tool_results


async def llm_answer(tool_results):
    """
    Generate an answer from the LLM based on the results of tool calls.
    """
    messages = cl.user_session.get("messages", []) or []
    messages.extend(tool_results)

    settings = cl.user_session.get("settings", {}) or {}

    stream = await openai_client.chat.completions.create(
        messages=messages,
        **settings,
        stream=True
    )

    answer_message: cl.Message = cl.user_session.get("answer_message")
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await answer_message.stream_token(token)

    await answer_message.update()
    messages.append({"role": "assistant", "content": answer_message.content})
    return answer_message


@cl.step(name="RAG Agent", type="run")
async def rag_agent(question):
    """
    Coordinate the RAG agent flow to generate a response based on the user's question.
    """
    # Step 1 - Call LLM with tool: plan to use tool or give message.
    message = await llm_tool(question)

    # Potentially several calls to retrieve context.
    if not message.tool_calls:
        answer_message: cl.Message = cl.user_session.get("answer_message")
        answer_message.content = message.content
        await answer_message.update()
        return message.content

    # Step 2 - Run the tool calls.
    tool_results = await run_multiple(message.tool_calls)

    # Step 3 - Call LLM to answer based on contexts (streamed).
    return (await llm_answer(tool_results)).content


@cl.on_chat_start
async def on_chat_start():
    """
    Send a welcome message and set up the initial user session on chat start.
    """
    await cl.Message(
        content="Welcome, please ask me anything about the Literal documentation!",
        disable_feedback=True
    ).send()

    cl.user_session.set("messages", prompt.format_messages())
    cl.user_session.set("settings", prompt.settings)
    cl.user_session.set("tools", prompt.tools)


@cl.on_message
async def main(message: cl.Message):
    """
    Main message handler for incoming user messages.
    """
    
    answer_message = cl.Message(content="")
    await answer_message.send()
    cl.user_session.set("answer_message", answer_message)

    await rag_agent(message.content)
