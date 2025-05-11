
import requests
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate



# Url
TARGET_URL = "https://ajuntament.barcelona.cat/novaciutadania/es/empadronamiento"

# Request and save html
response = requests.get(TARGET_URL)
html_content = response.text
with open("empadronamiento.html", "w", encoding="utf-8") as f:
    f.write(html_content)

# Load html using UnstructuredHTMLLoader
loader = UnstructuredHTMLLoader("empadronamiento.html")
docs = loader.load()
# Divide in chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n", ".", " "])
chunks = text_splitter.split_documents(docs)

# Embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Vectorial base
client = QdrantClient(host="localhost", port=6333)

collection_name = "migracion_espana"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    vectorstore = QdrantVectorStore.from_documents(documents=chunks,
        embedding=embedding_model,
        collection_name=collection_name,
        url="http://localhost:6333"
    )

else:
    print("🟡 La colección ya existe. No se reindexa.")

    vectorstore = QdrantVectorStore.from_existing_collection(
        collection_name="migracion_espana",
        embedding=embedding_model,
        url="http://localhost:6333",
    )

# LLMs
llm = OllamaLLM(model="mistral")

template = """
Responde a la pregunta del usuario usando la información a continuación. 
Usa únicamente español en la respuesta.

Contexto:
{context}

Pregunta:
{question}
"""

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)


# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=vectorstore.as_retriever(search_type="similarity", k=3),
#     return_source_documents=True
# )

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

query = "¿Qué necesito para empadronarme?"
result = qa_chain.invoke({"query": query})

print("🤖 Respuesta:", result["result"])
print("📄 Chunks usados:")
for doc in result["source_documents"]:
    print("-", doc.metadata)