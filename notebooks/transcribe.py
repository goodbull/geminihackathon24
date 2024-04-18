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
from youtube_transcript_api import YouTubeTranscriptApi
import re

class YoutubeVideoTranscriberSchema(BaseModel):
    """Inputs to the YoutubeVideoTranscriber"""
    url: str = Field(
        description="Youtube Video URL or Video ID"
    )

class YoutubeVideoTranscriberTool(BaseTool):
    """Custom Youtube Video Transcriber Tool to be callable by an Agent"""
    name = "Youtube_Video_Transcriber"
    description = "useful for when you need to transcribe a Youtube video given as a valid Youtube URL"
    args_schema: Type[BaseModel] = YoutubeVideoTranscriberSchema
    
    
    def get_clean_transcript(self, video_url):
    """Takes a youtube video id (in the video link, can look like: YrVVXFMgXrw). 
        Pulls transcript and returns clean   version."""
        # checks to see if full link then returns just id
        if "v=" in video_id:
            video_id = parse_video_id(video_url)
        # Pull transcript
        script = YouTubeTranscriptApi.get_transcript(video_url)
        # Combine 'text' fields into one string
        input_string = ' '.join([line['text'] for line in script])
        # remove special charaters
        pattern = re.compile(r'[^a-zA-Z0-9\s.,;:!"\'()\-—–?‘’“”\[\]]')
        transcript =  pattern.sub('', input_string)

        return transcript
    
    def parse_video_id(self, video_url):
        try:
            # Split the URL by "v="
            split_url = video_url.split("v=")
            # Extract the substring after "v="
            video_id = split_url[1]
            return video_id
        except IndexError:
            print("Invalid URL format")
            return None

    def _run(
        self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the Youtube Transcriber Tool."""
        try: 
            transcript = self.get_clean_transcript(url)
        except Exception as e:
            print(f"Error getting transcript from Youtube trying Cloud Speech API")
            ##TODO: Vinny to add the cloud speech api plugin
        return transcript

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


    