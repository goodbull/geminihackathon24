#Structured Topic Detection Tool
#Makes use of Vertex AI, Gemini 1.5 Pro to Detect Common topics in a video's frames and transcript using BERTopic 
#Returns: List[str]: a collection of topics
from google.cloud import aiplatform
import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_vertexai import VertexAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_google_vertexai import ChatVertexAI
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel, Part
from video_chunking import VideoPreprocessor
from moviepy.editor import VideoFileClip
from time import sleep
from task_configs import GEMINI_1_5_VIDEO_PROMPT, VIDEO_PART_MAX_DURATION
aiplatform.init(project="vertexai-gemini-hackathon-2024")
vertexai.init(project="vertexai-gemini-hackathon-2024")
vertexai.preview.init()

#video_url = "https://www.youtube.com/watch?v=f_lRdkH_QoY"
#video_downloader = VideoPreprocessor(output_directory="videos/")
#video_downloader.download_youtube_video(video_url)

video_uri = "gs://youtube_videos_1/Charlie Chaplin - The Kid - Fight Scene.mp4"
video_file = Part.from_uri(video_uri, mime_type="video/mp4")
# Safety config
safety_config = [
    generative_models.SafetySetting(
        category=generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
    ),
    generative_models.SafetySetting(
        category=generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
    ),
]

generative_multimodal_model = GenerativeModel("gemini-1.5-pro-preview-0409")
contents = [video_file, GEMINI_1_5_VIDEO_PROMPT]

response = generative_multimodal_model.generate_content(contents)
print(response)

""" print(response)

class AnswerWithJustification(BaseModel):
    '''An answer to the user question along with justification for the answer.'''
    answer: str
    justification: str

dict_schema = convert_to_openai_function(AnswerWithJustification)
print(dict_schema)
llm = ChatVertexAI(model_name="gemini-1.0-pro-001",max_retries=0, temperature=0)
structured_llm = llm.with_structured_output(dict_schema)

res = structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers ?")
print(res) """