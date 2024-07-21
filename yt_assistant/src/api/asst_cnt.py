from fastapi import Request
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response


async def getContent(request: Request):
    data = await request.json()
    yt_url, user_query = data.get("yt_url"), data.get("user_query")
    db = create_db_from_youtube_video_url(yt_url)
    res = get_response_from_query(db=db,query=user_query)
    return {
        "response" : res
    }
