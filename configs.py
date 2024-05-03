VIDEO_PART_MAX_DURATION=120 #Part duration in seconds
GEMINI_1_5_VIDEO_PROMPT=""" You are provided with a video or video transcript. Analyze the video or transcript and provide the following inform in a valid json format:
1. description: <a proper description of the video in 512 tokens>
2. key_topics: <a collection of 10 key topics as a valid python list>
3. sentiments: <a collection of 10 key sentiments as a valid python list>
4. emotions: <a collection of 10 key emotions as a valid python list>
5. people: <a list of all people mentioned in the video as a valid python list>
6. locations: <a list of all locations mentioned in the video as a valid python list>
7. age_appropriateness: <What age is the video appropriate for?>
8. movie_content_warnings: <a collection of all content warnings applicable to the video?
Make sure you generate the final output in a valid json format.
"""