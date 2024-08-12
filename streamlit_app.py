import streamlit as st
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
import shutil
import os
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate


CHROMA_PATH = "chroma"
# Load environment variables from .env file
load_dotenv()
# Access the environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
loader = CSVLoader(file_path="./FoodDataset.csv")
data = loader.load()

# CREATE CHUNKS USING TEXT SPLITTER BY LANGCHAIN
from langchain.schema.document import Document
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


# POPULATIN CHUNKED DATA TO CHROMA DB

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001" , google_api_key=GOOGLE_API_KEY)
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

# query db
PROMPT_TEMPLATE = """
provide recipe names, ingredients, total time to prepare,cusine, course and diet based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    # Prepare the DB.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY))
    retriever = db.as_retriever()
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=GOOGLE_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(model, retriever=retriever)

    response_text = qa_chain.run(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return response_text

def main():
    clear_database()
    chunks = split_documents(data)
    add_to_chroma(data)

    st.header("AI clone chatbot💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")
    response = query_rag(user_question)
    st.write("Reply: ", response)

   
if __name__ == "__main__":
    main()
