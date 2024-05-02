from langchain_google_vertexai import ChatVertexAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function

class TopicDetectionOutput(BaseModel):
    topics: list[str] = Field(description="A list of detected topics")

dict_schema = convert_to_openai_function(TopicDetectionOutput)

def detect_topics(text, num_topics=5):
    """
    Detects topics in the given text using ChatVertexAI.

    Args:
        text (str): The input text for topic detection.
        num_topics (int): The number of topics to detect (default: 5).

    Returns:
        list: A list of detected topics.
    """
    llm = ChatVertexAI(model_name="gemini-1.5-pro-latest", max_retries=0, temperature=0)
    structured_llm = llm.with_structured_output(dict_schema)

    prompt_template = PromptTemplate(
        input_variables=["text", "num_topics"],
        template="""
        Given the following text, detect {num_topics} main topics and return them as a list of strings.

        Text:
        {text}
        """,
    )

    prompt = prompt_template.format(text=text, num_topics=num_topics)
    response = structured_llm.invoke(prompt)
    topics = response.topics

    return topics

if __name__ == '__main__':
    text = "I am feeling Elated!"
    topics = detect_topics(text)
    print(topics)