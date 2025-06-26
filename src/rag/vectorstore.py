import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from unstructured.embed.openai import OpenAIEmbeddingEncoder, OpenAIEmbeddingConfig
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings

def embed_elements(elements, api_key: str):
    encoder = OpenAIEmbeddingEncoder(
        config=OpenAIEmbeddingConfig(
            model_name="text-embedding-3-small",
            api_key=api_key
        )
    )
    return encoder.embed_documents(elements)

def create_qdrant_client(api_key: str):
    return QdrantClient(
        url="https://01f14d7a-9e05-4804-bfd3-a69b4c3a97da.us-west-1-0.aws.cloud.qdrant.io:6333",
        api_key=api_key,
    )

def prepare_points(elements):
    points = []
    for element in elements:
        if element.embeddings:
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=element.embeddings,
                payload={
                    "text": element.text,
                    "metadata": element.metadata.to_dict()
                }
            ))
    return points

def upsert_to_qdrant(client, collection_name: str, points):
    if collection_name not in [col.name for col in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(points[0].vector), distance=Distance.COSINE)
        )
    client.upsert(collection_name=collection_name, points=points)

def get_langchain_vectorstore(client, collection_name: str):
    return Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        content_payload_key="text"
    )

# def retrieve_similar_chunks(query: str, k: int = 3) -> List[str]:
#     qdrant_store = Qdrant(
#         client=qdrant,
#         collection_name=COLLECTION_NAME,
#         embedding_function=embedding_model
#     )
#     results = qdrant_store.similarity_search(query, k=k)
#     return [r.page_content for r in results]
