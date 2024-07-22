from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI

class AIResponse(BaseModel):
    response: str = Field(description="Response from AI")
    highlights: List[str] = Field(description="highlights from summary")