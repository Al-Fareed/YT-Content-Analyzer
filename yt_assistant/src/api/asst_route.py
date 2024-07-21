from fastapi import APIRouter, Request
from yt_assistant.src.api import asst_cnt

router = APIRouter()

@router.post('/asst')
async def get_yt_asst(request: Request):
    return await asst_cnt.getContent(request)

