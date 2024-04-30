import os
import json
import asyncio
import chainlit as cl

from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec  # type: ignore

from literalai import AsyncLiteralClient

from dotenv import load_dotenv

load_dotenv()

client = AsyncLiteralClient()
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

pinecone_client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_spec = ServerlessSpec(
    cloud=os.environ.get("PINECONE_CLOUD"),
    region=os.environ.get("PINECONE_REGION"),
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
async def embed(question, model="text-embedding-ada-002"):
    embedding = await openai_client.embeddings.create(input=question, model=model)

    return embedding.data[0].embedding


@cl.step(name="Retrieve", type="retrieval")
async def retrieve(embedding, top_k):
    if pinecone_index == None:
        raise Exception("Pinecone index not initialized")
    response = pinecone_index.query(
        vector=embedding, top_k=top_k, include_metadata=True
    )
    return response.to_dict()


@cl.step(name="Retrieval", type="tool")
async def get_relevant_documentation_chunks(question, top_k=5):
    embedding = await embed(question)

    retrieved_chunks = await retrieve(embedding, top_k)

    contexts = [match["metadata"]["text"] for match in retrieved_chunks["matches"]]

    return "\n".join(contexts)


async def llm(
    messages,
    leverage_tools=False,
):
    settings = cl.user_session.get("settings", {}) or {}

    if leverage_tools:
        settings["tools"] = cl.user_session.get("tools")
        settings["tool_choice"] = "auto"

    response = await openai_client.chat.completions.create(
        messages=messages,
        **settings,
    )

    response_message = response.choices[0].message
    messages.append(response_message)
    return response_message


async def llm_tool(question):
    messages = cl.user_session.get("messages", []) or []
    messages.append({"role": "user", "content": question})

    settings = cl.user_session.get("settings", {}) or {}
    settings["tools"] = cl.user_session.get("tools")
    settings["tool_choice"] = "auto"

    response = await openai_client.chat.completions.create(
        messages=messages,
        **settings,
    )

    response_message = response.choices[0].message
    messages.append(response_message)
    return response_message

async def run_multiple(tool_calls):
    available_tools = {
        "get_relevant_documentation_chunks": get_relevant_documentation_chunks
    }

    async def run_single(tool_call):
        function_name = tool_call.function.name
        function_to_call = available_tools[function_name]
        function_args = json.loads(tool_call.function.arguments)

        # TODO: Parametrize the arguments depending on tool.
        function_response = await function_to_call(
            question=function_args.get("question"),
            top_k=function_args.get("top_k"),
        )
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response,
        }

    # Run tool calls in parallel.
    tool_results = await asyncio.gather(
        *(run_single(tool_call) for tool_call in tool_calls)
    )
    return tool_results


async def llm_answer(tool_results):
    """
    Call an LLM to answer the question based on tools results.
    """

    messages = cl.user_session.get("messages", []) or []
    messages.extend(tool_results)

    settings = cl.user_session.get("settings", {}) or {}
    settings["stream"] = True

    stream = await openai_client.chat.completions.create(
        messages=messages,
        **settings,
    )

    answer_message = cl.user_session.get("answer_message")

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await answer_message.stream_token(token)
    await answer_message.update()

    messages.append({"role": "assistant", "content": answer_message.content})
    return answer_message.content


@cl.step(name="RAG Agent", type="run")
async def rag_agent(question) -> str:
    """
    The RAG agent flow is:
    1. Call LLM to plan based on user question + tools available (context retrieval only for now)
    2. Run the tool calls, if any in plan
    3. Give tools results to LLM for pretty-print answer
    """

    # Step 1 - Call LLM with tool: plan to use tool or give message.
    message = await llm_tool(question)

    # Potentially several calls to retrieve context.
    if not message.tool_calls:
        return message.content

    # Step 2 - Run the tool calls.
    tool_results = await run_multiple(message.tool_calls)

    # Step 3 - Call LLM to answer based on contexts (streamed).
    return await llm_answer(tool_results)


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="Welcome, please ask me anything about the Literal documentation !"
    ).send()


    # We load the RAG prompt in Literal to track prompt iteration and
    # enable LLM replays from Literal AI.
    prompt_path = os.path.join(os.getcwd(), "./app/prompts/rag.json")
    with open(prompt_path, "r") as f:
        rag_prompt = json.load(f)

        prompt = await client.api.get_or_create_prompt(
            name=rag_prompt["name"],
            template_messages=rag_prompt["template_messages"],
            settings=rag_prompt["settings"],
            tools=rag_prompt["tools"],
        )

    cl.user_session.set("messages", prompt.format_messages())
    cl.user_session.set("settings", prompt.settings)
    cl.user_session.set("tools", prompt.tools)


@cl.on_message
async def main(message: cl.Message):
    # Hack to have agent logic part of Chatbot answer.
    answer_message = cl.Message(content="")
    await answer_message.send()
    cl.user_session.set("answer_message", answer_message)

    await rag_agent(message.content)
