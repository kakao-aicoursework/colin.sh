import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
import numpy as np

CHROMA_PERSIST_DIR = "./chroma"
CHROMA_COLLECTION_NAME = "kakao-service"

def init_db(dir_path: str):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            service_name, _ = os.path.splitext(file)
            docs = load_documents_from_file(file_path, service_name)
            save(docs)


def load_documents_from_file(file_path, service_name):
    documents = TextLoader(file_path).load()
    headers_to_split_on = [
        ("#", "title"),
    ]

    markdown_documents_with_meta = []

    for doc in documents:
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        markdown_documents = markdown_splitter.split_text(doc.page_content)
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=40)
        markdown_documents_split = text_splitter.split_documents(markdown_documents)

        # Add service name as meta-data to each document
        for split_doc in markdown_documents_split:
            split_doc.metadata['service'] = service_name
            print(split_doc.metadata)
            # setattr(split_doc, "service", service_name)  # use setattr to add attribute
            markdown_documents_with_meta.append(split_doc)

    return markdown_documents_with_meta

def save(documents):
    Chroma.from_documents(
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