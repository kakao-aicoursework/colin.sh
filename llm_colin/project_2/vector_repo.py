from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

def init_db(file: str):
    with open(file) as f:
        text = f.read()
    docs = embeddings(text, 500)
    return docs

def embeddings(text: str, chunk_size: int):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    splitted_text = splitter.split_text(text)

    embeddings = OpenAIEmbeddings()

    doc_search = Chroma.from_texts(
        splitted_text, embeddings, metadatas=[{"source": str(i)} for i in range(len(splitted_text))]
    )

    return doc_search

def get_relevant_documents(docs, number: int, user_input: str):
    return docs.as_retriever(search_kwargs={'k': number}).get_relevant_documents(user_input)