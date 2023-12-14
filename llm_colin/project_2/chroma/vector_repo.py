import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

CHROMA_PERSIST_DIR = "./chroma"
CHROMA_COLLECTION_NAME = "kakao-service"

def init_db(dir_path: str):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path) as f:
                text = f.read()
                docs = load_documents_from_file(text)
                save(docs)

def load_documents_from_file(documents):
    return split_by_chunk(documents)

def split_by_chunk(documents):
    text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
    return text_splitter.split_text(documents)

def save(documents):
    Chroma.from_texts(
        documents,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print('db save success')

def search(query, size):
    db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
    )
    docs = db.similarity_search(query=query, k=size)
    return docs