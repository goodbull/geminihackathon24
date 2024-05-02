from google.cloud import aiplatform
import vertexai
VIDEO_PART_MAX_DURATION=120 #Part duration in seconds
GEMINI_1_5_VIDEO_PROMPT=""" You are provided with a video. Your task is to analyze the video and extract the following information from the video:
1. description: <a proper description of the video in 512 tokens>
2. key_topics: <a collection of 10 key topics as a valid python list>
3. sentiments: <a collection of 10 key sentiments as a valid python list>
4. emotions: <a collection of 10 key emotions as a valid python list>
5. people: <a list of all people mentioned in the video as a valid python list>
6. locations: <a list of all locations mentioned in the video as a valid python list>
7. age_appropriateness: <What age is the video appropriate for?>
8. movie_content_warnings: <a collection of all content warnings applicable to the video?>
9. languages: <a collection of key languages spoken in the video>
10. main_language: <the main or dominant language spoken in the video>
The information must be extracted only from the video and you must not refer to outside sources. 
Make sure you generate the final output in a valid JSON format enclosed within opening and closing curly braces. Do not add the word JSON to the beginning of your response.
"""
#Generation Config
GEMINI_GENERATION_CONFIG = {
  "temperature": 0
}
# Safety config
GEMINI_SAFETY_SETTINGS = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]
aiplatform.init(project="vertexai-gemini-hackathon-2024")
vertexai.init(project="vertexai-gemini-hackathon-2024")
vertexai.preview.init()
DEFAULT_VIDEOS_BUCKET="youtube_videos_1"
DEFAULT_GEMINI_BATCH_SIZE=10
#increases 2x after each rate limit error encountered during inference calls
DEFAULT_BATCH_PROCESSING_DELAY=30#seconds 
GEMINI_1_5_PRO_SUPPORTED_AUDIO_MIME_TYPES=["audio/wav", "audio/mp3", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac"]
GEMINI_1_5_PRO_SUPPORTED_AUDIO_FORMATS = ["wav", "mp3", "aiff", "aac", "ogg", "flac"]