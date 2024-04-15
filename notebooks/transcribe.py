"""
Performs a transcription of a Youtube Video
Author: Pradeep Mohan
Date: 04/15/2024
"""
from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.tools import BaseTool, StructuredTool, tool


class YoutubeVideoTranscriberSchema(BaseModel):
    """Inputs to the YoutubeVideoTranscriber"""
    url: str = Field(
        description="Youtube Video URL."
    )

class YoutubeVideoTranscriberTool(BaseTool):
    """Custom Youtube Video Transcriber Tool to be callable by an Agent"""
    name = "Youtube_Video_Transcriber"
    description = "useful for when you need to transcribe a Youtube video given as a valid Youtube URL"
    args_schema: Type[BaseModel] = YoutubeVideoTranscriberSchema

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the Youtube Transcriber Tool."""
        transcript = """ EMPTY """
        return transcript

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


    