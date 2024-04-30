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
async def embed(query, model="text-embedding-ada-002"):
    embedding = await openai_client.embeddings.create(input=query, model=model)

    return embedding.data[0].embedding


@cl.step(name="Retrieve", type="retrieval")
async def retrieve(embedding):
    if pinecone_index == None:
        raise Exception("Pinecone index not initialized")
    response = pinecone_index.query(vector=embedding, top_k=5, include_metadata=True)
    return response.to_dict()


@cl.step(type="tool")
async def get_relevant_documentation_chunks(query, number_of_chunks=5):
    embedding = await embed(query)

    stored_embeddings = await retrieve(embedding)

    contexts = []

    for match in stored_embeddings["matches"]:
        contexts.append(match["metadata"]["text"])
    return "\n".join(contexts)


async def llm(
    messages,
    leverage_tools=False,
):
    settings = cl.user_session.get("settings", {}) or {}

    if leverage_tools:
        settings["tools"] = cl.user_session.get("tools")
        settings["tool_choice"] = "auto"
    print(settings)
    # TODO: Add streaming with tools.
    # TODO: If create is called with tools, make tool_choice="auto"?
    response = await openai_client.chat.completions.create(
        messages=messages,
        **settings,
        # tools=cl.user_session.get("tools"),
    )

    response_message = response.choices[0].message
    messages.append(response_message)
    cl.user_session.set("messages", messages)
    return response_message


async def plan_steps(question):
    messages = cl.user_session.get("messages", []) or []
    messages.append({"role": "user", "content": question})
    cl.user_session.set("messages", messages)

    return await llm(messages, leverage_tools=True)


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
            query=function_args.get("query"),
            number_of_chunks=function_args.get("number_of_chunks"),
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


async def pretty_print_answer(tool_results):
    """
    Call an LLM to answer the question based on tools results.
    """

    # TODO: Implement a cl.user_session.add("messages", msg)
    #       to avoid get/set pattern.
    messages = cl.user_session.get("messages", []) or []
    messages.extend(tool_results)
    cl.user_session.set("messages", messages)

    return (await llm(messages)).content


@cl.step(name="RAG Agent", type="run")
async def rag_agent(question) -> str:
    """
    The RAG agent flow is:
    1. Call LLM to plan based on user question + tools available (context retrieval only for now)
    2. Run the tool calls, if any in plan
    3. Give tools results to LLM for pretty-print answer
    """

    # Step 1 - Call LLM to plan steps.
    message = await plan_steps(question)

    if not message.tool_calls:
        return message.content

    # Step 2 - Run the tool calls.
    tool_results = await run_multiple(message.tool_calls)

    # Step 3 - Call LLM to pretty-print the answer.
    return await pretty_print_answer(tool_results)


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="Welcome, please ask me anything about the Literal documentation !"
    ).send()
    

    # TODO: Create the prompt only if the lineage doesn't exist.
    with open("./app/prompts/rag.json", "r") as f:
        rag_prompt = json.load(f)
        # TODO: Do a from_dict to create prompt
        # TODO: If no settings given, the default should be set no?
        #       Plus, the settings should be returned
        prompt = await client.api.get_or_create_prompt(
            name=rag_prompt["name"],
            template_messages=rag_prompt["template_messages"],
            settings=rag_prompt["settings"],
            tools=rag_prompt["tools"],
        )

    # prompt = await client.api.get_prompt(name="RAG prompt - Tooled")

    cl.user_session.set("messages", prompt.format_messages())
    cl.user_session.set("settings", prompt.settings)
    cl.user_session.set("tools", prompt.tools)


@cl.on_message
async def main(message: cl.Message):
    """
    Default assistant name is "Chatbot", can be changed in config.toml
    """

    # TODO: Move small hack to Chainlit code.
    #       (hack is to get the spinning wheel on assistant message)
    msg = cl.Message(content="")
    await msg.send()

    msg.content = await rag_agent(message.content)
    await msg.update()
