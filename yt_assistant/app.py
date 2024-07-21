from yt_assistant.src.api import asst_route
from fastapi import FastAPI
import uvicorn
app = FastAPI()

app.include_router(asst_route.router,prefix="/api")

@app.get('/')
async def check():
    return { "message":"Hello World"}


def start():
    uvicorn.run("yt_assistant.app:app", host="0.0.0.0", port=8000, reload=True)