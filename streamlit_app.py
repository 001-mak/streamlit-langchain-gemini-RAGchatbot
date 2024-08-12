import streamlit as st
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import shutil
import os
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS

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
    db = FAISS.from_documents(
        chunks, 
        embeddings, 
    )
    db.save_local("faiss_index")

def clear_database():
    if os.path.exists('./faiss_index'):
        shutil.rmtree('./faiss_index')

# query db
PROMPT_TEMPLATE = """
Answer the question based on the above context
format the recipe in recipe name, ingredients, total time to prepare,cusine, course and diet based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
def query_rag(query_text: str):
    # Prepare the DB.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001" , google_api_key=GOOGLE_API_KEY)
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=GOOGLE_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(model, retriever=retriever)

    response_text = qa_chain.run(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(response_text)
    return response_text

def main():
    clear_database()
    chunks = split_documents(data)
    add_to_chroma(data)

    st.header("Recipe chatbot👩🏻‍🍳")

    user_question = st.text_input("Ask for delicious recipes", key="user_question")
    response = query_rag(user_question)
    st.write(response)

   
if __name__ == "__main__":
    main()
