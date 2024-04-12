import os
import re
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

import pickle

load_dotenv()

documentation_path = os.path.join(os.getcwd(), "./documentation")

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

chroma_client = chromadb.PersistentClient(path=f"{os.getcwd()}/chromadb")


def create_dataset(id, path):
    values = []

    for filename in os.listdir(path):
        # Skip non mdx files
        if not filename.endswith("mdx"):
            continue

        with open(os.path.join(path, filename), "r") as file:
            file_content = file.read()

        # Extract file metadata
        metadata = re.search(
            r"---\ntitle: (.+?)(?:\ndescription: (.+?))?\n---", file_content, re.DOTALL
        )

        # Skip if missing no metadata
        if not metadata:
            continue
        file_title, file_description = metadata.groups()

        # Strop newlines except for code blocks
        content = file_content.split("---", 2)[-1]
        content_parts = re.split(r"(```.*?```)", content, flags=re.DOTALL)

        for i in range(len(content_parts)):
            if not content_parts[i].startswith("```"):
                content_parts[i] = content_parts[i].replace("\n", " ")
        content = "".join(content_parts)

        values.append(
            {"title": file_title, "description": file_description, "content": content}
        )

    return {"id": id, "values": values}


def create_embedding_set(dataset, model="text-embedding-ada-002"):
    values = []

    for item in dataset["values"]:
        input = f"title:{item['title']}_description:{item['description']}_content:{item['content']}"
        response = (
            openai_client.embeddings.create(input=input, model=model).data[0].embedding
        )
        values.append({"title": item["title"], "text": input, "values": response})

    return {"id": dataset["id"], "values": values}


def create_chroma_collection(name):
    if name in [collection.name for collection in chroma_client.list_collections()]:
        chroma_client.delete_collection(name)

    chroma_client.create_collection(name)
    return chroma_client.get_collection(name)


def upload_to_collection(collection, embedding_set, batch_size=100):
    values = embedding_set["values"]
    total_values = len(values)

    for i in range(0, total_values, batch_size):
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for j in range(i, min(i + batch_size, total_values)):
            ids.append(f"vector_{j}")
            embeddings.append(values[j]["values"])
            metadatas.append(
                {"dataset_id": embedding_set["id"], "text": values[j]["text"]}
            )
            documents.append(values[j]["title"])

        collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )


"""
    Uncomment the following lines to generate embeddings.
    Split the doc into chunks and calls OpenAI API to generate embeddings.

    Serializes the embeddings to a file for future use.
"""

embeddings = None
embeddings_filename = f"{os.getcwd()}/embed-documentation/literal-docs.emb"

if not os.path.exists(embeddings_filename):
    dataset = create_dataset("dataset_1", documentation_path)
    print("Dataset created from documentation")

    embeddings = create_embedding_set(dataset)
    print("Embeddings created from dataset")

    with open(embeddings_filename, "wb") as f:
        pickle.dump(embeddings, f)
    print("Embeddings saved to file for future use")
else:
    with open(embeddings_filename, "rb") as f:
        embeddings = pickle.load(f)
    print("Embeddings loaded from file")

chroma_collection = create_chroma_collection("literal-doc-collection")
print("Chroma collection created")

upload_to_collection(chroma_collection, embeddings)
print("Embeddings uploaded to collection")
