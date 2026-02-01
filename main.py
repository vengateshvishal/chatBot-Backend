import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi.middleware.cors import CORSMiddleware

# 1. Initialize FastAPI
app = FastAPI(title="HR Policy Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (POST, GET, etc.)
    allow_headers=["*"], # Allows all headers
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"

loader = PyPDFLoader("HR_Policy_Sample.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url=BASE_URL,
    api_key=API_KEY,
)

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0,
    openai_api_key=API_KEY,
    base_url=BASE_URL
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer ONLY from the context below.
If not found, say "I don't know".

Context:
{context}

Question:
{question}
""")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        result = rag_chain.invoke(request.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
