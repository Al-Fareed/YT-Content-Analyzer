from yt_assistant.src.api.res_model import AIResponse
from fastapi import Request
from langchain_community.document_loaders import YoutubeLoader
from langchain.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
parser = PydanticOutputParser(pydantic_object=AIResponse)

class Embeddings:
    def __init__(self):
        self.embeddings_model = "text-embedding-3-small"
        self.embedding_instance = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model=self.embeddings_model)

    def get_vector_db_instance(self, docs):
        return FAISS.from_documents(documents=docs, embedding=self.embedding_instance)



def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    embedding_instance = Embeddings()

    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = embedding_instance.get_vector_db_instance(docs)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a expert in Q&A, whose main task or goal is to answer the questions posed by the user on the specific youtube video.
        You will receive the youtube video transcript as INPUT.
        You will give out the 
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}

        Mandatory Instructions:
        1. Only use the factual information from the transcript to answer the question.
        2. If you feel like you don't have enough information to answer the question, say "I don't know".
        3. Your answers should be verbose and detailed, 

        Steps for you to follow before giving out the response in the derived format:
        1. Understand the query from the user
        2. Identify which part of the video to derive out the answer for the user query
        3. Derive out the summary in your mind
        4. Finally format it in the given json as below
            ```
            {{
                "summary": "",
                "highlights": [<important highlights of the above summary>]
            }}
        """
    )

    chain = prompt | llm

    response = chain.invoke({"question": query, "docs": docs_page_content})
    response_content = response.content
    response_json = json.loads(response_content)
    print(json.dumps(response_json, indent=4))
    return response_json


async def getContent(request: Request):
    data = await request.json()
    yt_url, user_query = data.get("yt_url"), data.get("user_query")
    db = create_db_from_youtube_video_url(yt_url)
    res = get_response_from_query(db=db,query=user_query)
    return {
        "response": res['summary'],
        "highlights": res['highlights']
     }
