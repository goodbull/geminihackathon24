from google.cloud import aiplatform
from langchain_google_vertexai import ChatVertexAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.messages import HumanMessage
import time
from vertexai.preview import generative_models
from vertexai.preview.generative_models import GenerativeModel, Part
from task_configs import (
    GEMINI_1_5_VIDEO_TRANSCRIPT_PROMPT,
    GEMINI_GENAI_OBJECT,
    GEMINI_GENERATION_CONFIG,
    GEMINI_SAFETY_SETTINGS

)
from benchmark_examples import (
    GEMINI_1_5_PRO_BENCHMARK_EXTRACTION_LEX_TUCKER_TRANSCRIPT
)

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

def detect_topics_sentiment(transcript_text):
    """
    Detects topics in the given text using ChatVertexAI.

    Args:
        text (str): The input text for topic detection.
    Returns:
        list: A list of detected topics.
    """
    
    generative_multimodal_model_vertex = GenerativeModel(model_name="gemini-1.0-pro-vision", 
                                                         generation_config=GEMINI_GENERATION_CONFIG)
    generative_multimodal_model = GEMINI_GENAI_OBJECT.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=GEMINI_GENERATION_CONFIG,safety_settings=GEMINI_SAFETY_SETTINGS)
    contents = [transcript_text, GEMINI_1_5_VIDEO_TRANSCRIPT_PROMPT]
    response = generative_multimodal_model.generate_content(contents)
    response_text = str(response.text)
    #print(response_text)
    text_json = {"type":"text",
                 "text":response_text, 
               }
    text_message = {"type": "text",
                    "text": "What are the contents of the JSON ?",
                    }
    
    message_contents =  [text_message,text_json]   
    message = HumanMessage(content=message_contents)
    llm = ChatVertexAI(model_name="gemini-1.0-pro-001",max_retries=0, temperature=0)
    structured_llm = llm.with_structured_output(dict_schema)
    extraction_response_dict = structured_llm.invoke([message])[0]
    extraction_response = extraction_response_dict['args']
    return extraction_response

if __name__ == '__main__':
    transcript_text = GEMINI_1_5_PRO_BENCHMARK_EXTRACTION_LEX_TUCKER_TRANSCRIPT
    topics_sentiment = detect_topics_sentiment(transcript_text)
    print(topics_sentiment)