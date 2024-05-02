from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List, Dict, Type, Optional
from vertexai.preview.generative_models import GenerativeModel, Part
from google.cloud import storage
import traceback
import requests

class GeminiProVisionVideoInfoExtractionSchema(BaseModel):
    """Inputs to the GeminiProVisionVideoInfoExtractor"""
    url: Optional[str] = Field(description="Youtube Video URL or Video ID")
    uri: Optional[str] = Field(description="Video URI inside a Google Cloud Storage Bucket")
    
    @validator("url","uri")
    @classmethod
    def valid_url_or_uri(cls,field_values):
        """Checks if the URI path to an object is a valid one"""
        def check_uri_object_exists(uri: str):
            storage_client = storage.Client()
            bucket = storage_client.bucket(uri.split('/')[2])
            objectname = None if len(uri.split('/')) < 4 else "/".join(uri.split("/")[3:])
            if objectname is None:
                raise ValueError("This appears to be a path to the parent folder of an object")
            if not bucket.blob(objectname).exists():
                raise ValueError(f"Appears that the specified object in the uri: {uri} doen't exist")
            return None
        
        if field_values["url"] is None and field_values["uri"] is None:
            raise ValueError("Both Video URL and URI cannot be empty")
        if field_values["uri"] is not None and not field_values["uri"].startswith("gs://") and field_values["url"] is None:
            raise ValueError("The Google Cloud Storage uri of the video must begin with gs://")
        else:
            try:
                check_uri_object_exists(field_values["uri"])
            except ValueError as v:
                traceback.print_exc()
                raise ValueError("uri field failed validation check, perhaps object in uri doesn't exist")
        if field_values["uri"] is not None and field_values["url"] is not None:
            raise ValueError(f"Ambiguous Specification, both, URL: {field_values["url"]} and URI: {field_values["uri"]} currently unsupported")
        if field_values["url"] is not None and field_values["uri"] is None:
            r = requests.get(field_values["url"]) 
            if "video unavailable" in r.text.lower():
                raise ValueError(f"Invalid Youtube Video URL, {field_values["url"]} specified")
            
            
class GeminiProVideoJsonStringParserSchema(BaseModel):
    """Inputs to the GeminiProVideoJSONStringParser"""
    jsons: List[str] = Field(description="A list of valid jsons as strings")
    
    @validator("json")
    @classmethod
    def valid_list_of_jsons_as_strings(cls,field_values):
        json_list = field_values["jsons"]
        valid_string = list(map(lambda x:isinstance(x,str),json_list))        
        if valid_string.count(False)>0:
            raise ValueError(f"Invalid JSON string found")
        valid_non_empty_string = list(map(lambda x:x not in [""," ","``` ```","``````"],json_list))
        if valid_non_empty_string.count(False)>0:
            raise ValueError(f"Empty JSON String Found")

                            
class YoutubeVideoTranscriberSchema(BaseModel):
    """Inputs to the YoutubeVideoTranscriber"""
    url: str = Field(
        description="Youtube Video URL or Video ID"
    )
    @validator("url")
    @classmethod
    def validate_youtube_video_url(cls,field_values):
        if field_values["url"] is not None:
            r = requests.get(field_values["url"]) 
            if "video unavailable" in r.text.lower():
                raise ValueError(f"Invalid Youtube Video URL, {field_values["url"]} specified")
        else:
            raise ValueError(f"Invalid URL string")
        
        

    
        
                                                                   
        

            
            
    