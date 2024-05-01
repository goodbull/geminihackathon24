import re
from youtube_transcript_api import YouTubeTranscriptApi

class TranscribeTranslate():
    """A solution to transcribe and translate youtube videos by just providing their links."""

    def parse_video_id(video_url):
        try:
            # Split the URL by "v="
            split_url = video_url.split("v=")
            # Extract the substring after "v="
            video_id = split_url[1]
            return video_id
        except IndexError:
            return "Invalid URL format"

    def clean_transcript(transcript):
        """Takes a youtube video id (in the video link, can look like: YrVVXFMgXrw).
        Pulls transcript and returns clean version."""
        input_string = ' '.join([line['text'] for line in transcript])
        # remove special charaters
        pattern = re.compile(r'[^a-zA-Z0-9\s.,;:!"\'()\-—–?‘’“”\[\]]')
        clean_transcript =  pattern.sub('', input_string)

        return clean_transcript

    def video_transcript_list(video_id):
        print(video_id)
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        return transcript_list

    def video_transcript(transcript_list, language_code = ["en"]):
        return transcript_list.find_transcript(language_code)

    def fetch_transcript(transcript):
        return transcript.fetch()

    def video_language(transcript):
        return transcript.language

    def video_translation_languages(transcript):
        return transcript.translation_languages

    def video_language_supported(translation_languages, language):
        count = 0
        nlang = len(translation_languages)
        for lang in translation_languages:
            count += 1
            if lang['language_code'] == language:
                print(f"Yes, language '{lang['language_code']}' is translatable.")
                return True
            else:
                if count == nlang:
                    # print(f"Language '{language}' is not supported for translation")
                    break
                else:
                    continue
        return False

    def video_translation(transcript, language):
        # whether it has been manually created or generated by YouTube
        f"Transcription generated by Youtube {transcript.is_generated}\n\n"
        # whether this transcript can be translated or not
        f"Is it translatable? {transcript.is_translatable}\n\n"
        # a list of languages the transcript can be translated to
        f"List of languages it can be translated {transcript.translation_languages}\n\n"

        if video_language_supported(video_translation_languages(transcript), language):
            language in transcript.translation_languages
            translation = transcript.translate(language)
            return fetch_transcript(translation)
        else:
            return [{'text': f"Language '{language}' is not available for translation.", 'start': 0.0, 'duration': 0.0}]
    