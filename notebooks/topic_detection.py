#Structured Topic Detection Tool
#Makes use of Vertex AI, Gemini 1.5 Pro to Detect Common topics in a video's frames and transcript using BERTopic 
#Returns: List[str]: a collection of topics
from google.cloud import aiplatform
import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict
from langchain_google_vertexai import VertexAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_core.utils.function_calling import convert_to_openai_function
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel, Part
from video_chunking import VideoPreprocessor
from moviepy.editor import VideoFileClip
from langchain_core.messages import HumanMessage
from time import sleep
from task_configs import GEMINI_1_5_VIDEO_PROMPT, VIDEO_PART_MAX_DURATION
import json
import traceback

aiplatform.init(project="vertexai-gemini-hackathon-2024")
vertexai.init(project="vertexai-gemini-hackathon-2024")
vertexai.preview.init()


#video_url = "https://www.youtube.com/watch?v=f_lRdkH_QoY"
#video_downloader = VideoPreprocessor(output_directory="videos/")
#video_downloader.download_youtube_video(video_url)

video_uri = "gs://youtube_videos_1/Charlie Chaplin - The Kid - Fight Scene.mp4"
video_file = Part.from_uri(video_uri, mime_type="video/mp4")
generation_config = {
  "temperature": 0
}
# Safety config
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]

generative_multimodal_model = GenerativeModel(model_name="gemini-1.0-pro-vision",generation_config=generation_config)
contents = [video_file, GEMINI_1_5_VIDEO_PROMPT]

response = generative_multimodal_model.generate_content(contents)
response_text = str(response.text)
response_text = response_text.split("json")[1]
response_text = response_text.split("```")[0]
try:
    response_text_dict = json.loads(response_text)
except Exception as e:
    traceback.print_exc()

print(response_text_dict)


##Running using Langchain
class VideoTopicExtraction(BaseModel):
    '''An Extraction of Key Features from an input video JSON'''
    description: str = Field(description="A 512 token description of the contents of the video")
    key_topics: List[str] = Field(description="A collection of key topics.")
    sentiments: List[str] = Field(description="A collection of key sentiments expressed in the video")
    emotions: List[str] = Field(description="A collection of 10 key emotions expressed in teh video")
    people: List[str] = Field(description="A list of all people mentioned or featuring in the video")
    locations: List[str] = Field(descritpion="A list of all locations and landmarks referred or appearing i=in the video")
    age_appropriateness: str = Field(description="Age groups of audience the video is appropriate for")
    movie_content_warnings: List[str] = Field(description="a collection of all content warnings applicable to the video")
dict_schema = convert_to_openai_function(VideoTopicExtraction)
print(dict_schema)
image_message = {
    "type": "image_url",
    "video": {"video": video_uri},
}
text_message = {
    "type": "text",
    "text": "What are the contents of the JSON",
}
message = HumanMessage(content=[text_message, image_message])
llm = ChatVertexAI(model_name="gemini-1.0-pro-vision",max_retries=0, temperature=0)
structured_llm = llm.with_structured_output(dict_schema)
try:
    res = structured_llm.invoke([message])
    print(f"Response from Direct Structured LLM Call: {res}")
except Exception as e:
    traceback.print_exc()

