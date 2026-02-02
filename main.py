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
from pinecone import Pinecone ,ServerlessSpec


app = FastAPI(title="HR Policy Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"
PINECONE_API_KEY ="pcsk_5aPHPw_J19ZvwnzBa1NBsM9jEqxLDM6ikMbQwKwiSt4tE76XJw9NAbkX1xN5pRfvnK1KzD"

pdf_files =["HR_Policy_Sample.pdf","Resume(Vishal).pdf","module 5(2 mark).pdf"]
pc=Pinecone(api_key=PINECONE_API_KEY)
index_name="ChatBot"
index = pc.Index("quickstart")

documents=[]

for pdf in pdf_files:
    loader =PyPDFLoader(pdf)
    documents.extend(loader.load())

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
You must prepend the filename of the source document to your answer in the format: [filename]: [answer].
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
