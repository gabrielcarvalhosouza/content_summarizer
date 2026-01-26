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

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.genai import types
    from openai import OpenAI
    from openai.types.chat import ChatCompletion

    from content_summarizer.core import AppConfig
logger: logging.Logger = logging.getLogger(__name__)

GEMINI_MODEL_MAP = {
    "2.5-flash": "gemini-2.5-flash",
    "2.5-pro": "gemini-2.5-pro",
    "3-flash": "gemini-3-flash-preview",
    "3-pro": "gemini-3-pro-preview",
}

SYSTEM_PROMPT_TEMPLATE = """
You are a Senior Content Analyst, recognized for your ability to synthesize complex information into clear, dense, and actionable reports. Your task is to process a video transcript and generate a professional summary.

### Tone Guidelines
- **Strictly Professional:** Maintain a neutral, journalistic, and objective tone.
- **Objectivity:** Get straight to the point.
- **Impartiality:** Do not express opinions on the content; only report what was said with precision.

### Structure Guidelines
- **Flexible Structure:** There is no rigid section structure. Organize the summary in the way that best adapts to the original content, ensuring fluidity and coherence.
- **Central Thesis:** Ensure the central thesis of the video is clear right at the beginning.
- **Conciseness:** Be concise but comprehensive. Use up to **600 words** if the content's complexity requires it, but do not use filler content. If the video is short, be brief.
- **Anti-Ad Filter:** Strictly ignore interrupted sponsorship segments (e.g., "This video is brought to you by...", "Use my coupon"), excessive self-promotion, or engagement requests ("Smash the like button").
- **Ad Exception:** However, if the entire video is a dedicated review or analysis of a product/service, treat it as the main topic and summarize it impartially.

### Formatting Rules
- Use **Markdown** to format titles, bold text, and lists.
- **NEVER** use introductory phrases like "Here is the summary" or "The video discusses". Start directly with the title or the Overview.
- If the content is technical, preserve the correct terminology.
- **Output Language:** {user_language}.
"""  # noqa: E501


class SummaryError(Exception):
    """Custom exception for errors during the summary generation process."""

    pass


def _read_transcription(input_file_path: Path) -> str:
    if not input_file_path.exists():
        logger.error("Input file not found")
        raise FileNotFoundError("Input file not found")
    with input_file_path.open("r", encoding="utf-8") as f:
        return f.read()


def _summarize_gemini(
    config: AppConfig,
    input_file_path: Path,
) -> str | None:
    """Generate a summary from a text file using the Gemini API.

    This function reads a text file (like a transcription or caption), constructs a
    detailed prompt, sends it to the Gemini API, and returns the
    resulting summary.

    Args:
        config: Object containing all necessary settings for Gemini summarization.
        input_file_path: The path to the text file to be summarized.

    Returns:
        The generated summary text as a string, or None if the API
        returns no text.

    Raises:
        SummaryError: If the API call fails or another exception occurs.

    """
    from google import genai

    client: genai.Client = genai.Client(api_key=config.gemini_key)

    model_name: str = GEMINI_MODEL_MAP.get(
        config.gemini_model_name, config.gemini_model_name
    )

    transcription_content: str = _read_transcription(input_file_path)

    system_instructions: str = SYSTEM_PROMPT_TEMPLATE.format(
        user_language=config.user_language
    )
    final_prompt = f"{system_instructions}\n\n### Input Text:\n{transcription_content}"
    try:
        logger.info("Generating summary")
        res: types.GenerateContentResponse = client.models.generate_content(
            model=model_name,
            contents=final_prompt,
        )
        logger.info("Summary generated successfully")
        return res.text
    except Exception as e:
        logger.exception("Failed to generate summary")
        raise SummaryError("Failed to generate summary") from e


def _summarize_openai_compatible(
    config: AppConfig,
    input_file_path: Path,
) -> str | None:
    """Generate a summary from a text file using Ollama.

    This function reads a text file (like a transcription or caption), constructs a
    detailed prompt, sends it to the Ollama API, and returns the
    resulting summary.

    Args:
        config: Object containing all necessary settings for Ollama summarization.
        input_file_path: The path to the text file to be summarized.

    Returns:
        The generated summary text as a string, or None if the API
        returns no text.

    Raises:
        SummaryError: If the API call fails or another exception occurs.

    """
    from openai import OpenAI

    transcription_content: str = _read_transcription(input_file_path)

    system_instructions: str = SYSTEM_PROMPT_TEMPLATE.format(
        user_language=config.user_language
    )

    base_url: str = config.ollama_url.strip().rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"

    try:
        logger.info("Generating summary")
        client: OpenAI = OpenAI(base_url=base_url, api_key="ollama")
        res: ChatCompletion = client.chat.completions.create(
            model=config.ollama_model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_instructions,
                },
                {
                    "role": "user",
                    "content": transcription_content,
                },
            ],
            extra_body={"options": {"num_ctx": config.ollama_ctx}},
        )

        logger.info("Summary generated successfully")
        return res.choices[0].message.content
    except Exception as e:
        logger.exception("Failed to generate summary")
        raise SummaryError("Failed to generate summary") from e


def generate_summary(
    config: AppConfig,
    input_file_path: Path,
) -> str | None:
    """Dispatch the summary generation to the configured provider.

    Args:
        config: The application configuration object.
        input_file_path: The path to the text file to be summarized.

    Returns:
        The generated summary text as a string.

    Raises:
        SummaryError: If the provider is invalid or the generation fails.

    """
    if config.provider == "gemini":
        return _summarize_gemini(config, input_file_path)
    if config.provider == "ollama":
        return _summarize_openai_compatible(config, input_file_path)
    raise SummaryError(f"Unsupported provider: {config.provider}")
