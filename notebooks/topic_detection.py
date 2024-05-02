#Structured Topic Detection Tool
#Makes use of Vertex AI, Gemini 1.5 Pro to Detect Common topics in a video's frames and transcript using BERTopic 
#Returns: List[str]: a collection of topics
from google.cloud import aiplatform
import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from typing import List, Dict, Optional, Type
from langchain_google_vertexai import VertexAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_core.utils.function_calling import convert_to_openai_function
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel, Part
from video_chunking import VideoPreprocessor
from langchain_core.messages import HumanMessage
from tool_schemas import (
    GeminiProVisionVideoInfoExtractionSchema, 
    GeminiProVideoJsonStringParserSchema
)
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from time import sleep
from task_configs import (
GEMINI_1_5_VIDEO_PROMPT, 
VIDEO_PART_MAX_DURATION,
GEMINI_GENERATION_CONFIG,
GEMINI_SAFETY_SETTINGS,
DEFAULT_GEMINI_BATCH_SIZE,
DEFAULT_BATCH_PROCESSING_DELAY)
import json
import traceback
from more_itertools import chunked
import time


#video_url = "https://www.youtube.com/watch?v=f_lRdkH_QoY"

class GeminiProVisionVideoInfoExtractionTool(BaseTool):
    name = "geminipro_vision_video_info_extractor"
    description = "useful for when you have to extract information pertaining to a given video "
    args_schema: Type[BaseModel] =  GeminiProVisionVideoInfoExtractionSchema
    def gen_batches_of_uris(uri_list,batch_size=10):
        batches = []
        
        
    
    def _run(
        self, url: str, uri: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        if url is None and uri is not None:
            #Process uri, at this point it will be a valid uri of a Google Cloud Storage bucket
            video_file = Part.from_uri(uri, mime_type="video/mp4")
            generative_multimodal_model = GenerativeModel(model_name="gemini-1.5-pro-latest",
                                              generation_config=GEMINI_GENERATION_CONFIG)
            contents = [video_file, GEMINI_1_5_VIDEO_PROMPT]
            response = generative_multimodal_model.generate_content(contents)
            response_text = str(response.text)
        else if uri is None and url is not None:
            #Process url using Video Processor and get the default segments of the video
            video_chunker = VideoPreprocessor(output_directory="videos/")
            video_chunker.download_youtube_video(video_url)
            video_chunker.basic_segmentation()
            #Extract the key frames of the video
            video_chunker.extract_frames()
            #extract audio chunks
            video_chunker.extract_audio()
            #save the metadata
            video_chunker.save_metadata()            
            #Upload the Videos, the frames, the audio files and the transcripts to Cloud Storage
            cloud_storage_metadata_dict = video_chunker.upload_to_cloud_storage(bucket=DEFAULT_VIDEOS_BUCKET)
            #Get the uris of the video segments in cloud storage
            urilist = []
            #Create batches of uris to 
            
            
            
            
            
            
            
        return response_text, transcript

    async def _arun(
        self, url: str, uri: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


#video_uri = "gs://youtube_videos_1/Charlie Chaplin - The Kid - Fight Scene.mp4"
class GeminiProVideoJSONStringParserTool(BaseTool):
    name = "geminipro_video_json_parser_tool"
    description = "useful for when you have to parse a string representing a JSON schema to extract relevant information about a video and output as a structured and cleaner Python dictionary schema"
    args_schema: Type[BaseModel] = GeminiProVideoJsonStringParserSchema
    
    def _run(
        self, jsons: List[str],run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
    



##Running using Langchain
class VideoTopicExtraction(BaseModel):
    '''An Extraction of Key Features from an input video JSON'''
    description: str = Field(description="A 512 token description of the contents of the video")
    key_topics: List[str] = Field(description="A collection of key topics.")
    sentiments: List[str] = Field(description="A collection of key sentiments expressed in the video")
    emotions: List[str] = Field(description="A collection of key emotions expressed in the video")
    people: List[str] = Field(description="A list of all people mentioned or featuring in the video")
    locations: List[str] = Field(description="A list of all locations and landmarks referred or appearing  in the video")
    age_appropriateness: str = Field(description="Age groups of audience the video is appropriate for")
    movie_content_warnings: List[str] = Field(description="a collection of all content warnings applicable to the video")
    languages: List[str] = Field(description="A list of all languages spoken in the video")
    main_language: List[str] = Field(description="A the main language of the video")
    
dict_schema = convert_to_openai_function(VideoTopicExtraction)
text_message_json = text_message = {
    "type": "text",
    "text": response_text,
}  
text_message = {
    "type": "text",
    "text": "What are the contents of the JSON ?",
}
message = HumanMessage(content=[text_message, text_message_json])
llm = ChatVertexAI(model_name="gemini-1.0-pro-001",max_retries=0, temperature=0)
structured_llm = llm.with_structured_output(dict_schema)
try:
    res = structured_llm.batch([[message]])
    print(f"Response from Direct Structured LLM Call: {res}")
except Exception as e:
    traceback.print_exc()

