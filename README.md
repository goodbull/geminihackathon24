# Gemini Hackathon 2024
Deloitte Team Submission GCP/Gemini Hackathon. Showcasing a Youtube Multimodal analysis chrome plugin.

## Youtube Multimodal Video Analysis Chrome Plugin
## Team Members:
### Vinicus Granja
### Jay Park
### Pradeep Mohan
### Antoine
### Hector
### Andrew Irwin
### Trenten Babcock


# YouTube-Transcript-Summarizer

**Youtube Transcript Summarizer** is a Chrome Extension that allows users to get a **summarized** version of the transcripts of YouTube videos with a **single click**. It is built on a **Flask Backend REST API** to expose the summarization service to the client.


## Project Stages
![alt text](/extension/images/stages.png?raw=true)


## Features
- Allows the user to adjust the maximum length of the summarized text through dynamic truncation.
- Adopts a language-agnostic approach and supports transcript summarization even for videos without subtitles.
- Employs an asynchronous XMLHttpRequest to ensure non-blocking communication with the Flask Backend.


## Output Screenshot
![alt text](/extension/images/output.png?raw=true)

## Youtube Link to Demo
![alt text](https://www.youtube.com/watch?v=bu5BQ0PadBo)

## Installation
- Clone this repository to your local machine:
-
  ```
  git clone https://github.com/<your-username>/geminihackathon24.git

  cd YouTube-Transcript-Summarizer
  ```

- Create the conda environment using the `environment.yml` file: `conda env create -f environment.yml`
- Activate the conda environment: `conda activate hackathonEnv`
- Next, install the dependencies:
-
  ```
  pip install -r Requirements.txt
  ```

- Next create a .env file with your gemini API key `GOOGLE_API_KEY=<your-key-here>`
- To execute the Application locally:
  - Start the Flask backend on the terminal using the following command:

    ```
    python TranscriptApp.py
    ```
    This will start a local server at ```http://127.0.0.1:5000/```. You may see a couple of warnings but it's all good and you may ignore it!
  - Load the extension into Google Chrome:
    - Open Google Chrome and go to ```chrome://extensions/```.
    - Enable the ```Developer mode``` toggle in the top right corner.
    - Click on ```Load unpacked``` and then select the folder ```extension``` inside the directory folder ```geminihackathon24```. The one you cloned from this repository.
  - You should now see the extension loaded in the Chrome toolbar. Navigate to any YouTube video, click on the extension icon, and click "Summarize" to see the summary of the video   transcript.
  - All Done..!!


## Contribution
Contributions to this project are *welcome!* If you wish to contribute, please follow these steps:
- Fork the repository.
- Create a new branch for your features or fixes.
- Make your changes and commit them.
- Push your changes to your fork.
- Create a Pull Request from your fork to this repository.
- Make sure to update the ```Requirements.txt``` file if you've added any new dependencies.
