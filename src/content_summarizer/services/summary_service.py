"""Provides a service for generating summaries using the Gemini API.

This module contains the function responsible for communicating with the
Google Generative AI API, sending a transcription, and receiving a
generated summary. It encapsulates the prompt engineering and error
handling for this specific task.

"""
# Copyright 2025 Gabriel Carvalho
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import textwrap
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)

GEMINI_MODEL_MAP = {
    "1.0-pro": "models/gemini-1.0-pro",
    "1.5-flash": "models/gemini-1.5-flash-latest",
    "1.5-pro": "models/gemini-1.5-pro-latest",
    "2.5-flash": "models/gemini-2.5-flash",
    "2.5-pro": "models/gemini-2.5-pro",
}


class SummaryError(Exception):
    """Custom exception for errors during the summary generation process."""

    pass


def generate_summary(
    gemini_model_name: str,
    gemini_api_key: str,
    user_language: str,
    input_file_path: Path,
) -> str | None:
    """Generate a summary from a text file using the Gemini API.

    This function reads a text file (like a transcription or caption), constructs a
    detailed prompt, sends it to the Gemini API, and returns the
    resulting summary.

    Args:
        gemini_model_name: The name of the Gemini model to use.
        gemini_api_key: The API key for the Gemini service.
        user_language: The target language for the summary (e.g., 'en-US').
        input_file_path: The path to the text file to be summarized.

    Returns:
        The generated summary text as a string, or None if the API
        returns no text.

    Raises:
        FileNotFoundError: If the input_file_path does not exist.
        SummaryError: If the API call fails or another exception occurs.

    """
    import google.generativeai as genai
    from google.generativeai.generative_models import GenerativeModel
    from google.generativeai.types import GenerateContentResponse

    genai.configure(api_key=gemini_api_key)
    gemini_model: GenerativeModel = genai.GenerativeModel(
        GEMINI_MODEL_MAP[gemini_model_name]
    )

    if not input_file_path.exists():
        logger.error("Input file not found")
        raise FileNotFoundError("Input file not found")

    with input_file_path.open("r", encoding="utf-8") as f:
        transcription_content: str = f.read()
        prompt: str = textwrap.dedent(f"""
            You are an expert summarizer with a knack for clarity and a great sense of humor. Your mission is to distill the following video transcript into a summary that is natural, engaging, and easy to read, as if a friend were explaining the main points.

            Rules:

            Core Mission: Summarize all key points with clarity and objectivity. Capture the essence of the content.
            Formatting Freedom: Feel free to use bullet points, standard paragraphs, or a hybrid format—whichever presents the information most effectively and clearly.
            Word Count: Be as concise as possible, but you can go up to  1500 words if the content's complexity truly justifies it. No need to fill space unnecessarily.
            Match the Vibe: If the video is casual and humorous, reflect that with some clever wit, but keep the core information sharp. If the content is serious, dial back the jokes but maintain an engaging, non-robotic tone. A light, witty remark is fine even in serious topics.
            Be Seamless: Dive right into the summary. Do not use opening phrases like "This is a summary of..." or "The video discusses...".
            Output Language: The summary must be written in {user_language}. Always output in Markdown format. Ignore self-promosions or ads.
            Content: {transcription_content}
            """)  # noqa: E501
    try:
        logger.info("Generating summary")
        res: GenerateContentResponse = gemini_model.generate_content(prompt)
        logger.info("Summary generated successfully")
        return res.text
    except Exception as e:
        logger.exception("Failed to generate summary")
        raise SummaryError("Failed to generate summary") from e
