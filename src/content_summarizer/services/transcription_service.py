"""Provides services for audio transcription.

This module offers functions for transcribing audio files using either
a local Whisper model or a remote transcription API. It encapsulates
the logic for both methods, handling model loading, API requests, and
error handling.

Functions:
    fetch_transcription_local: Transcribes audio using a local model.
    fetch_transcription_api: Transcribes audio using a remote API.
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

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import IO

import requests

logger: logging.Logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Custom exception for errors during the transcription process."""

    pass


def fetch_transcription_local(
    audio_file_path: Path,
    whisper_model_name: str,
    beam_size: int,
    device: str,
    compute_type: str,
) -> str:
    """Transcribe an audio file locally using a Whisper model.

    This function loads a Whisper model and runs the transcription
    on the local machine.

    Args:
        audio_file_path: The path to the audio file to be transcribed.
        whisper_model_name: The name of the Whisper model to use.
        beam_size: The beam size for the transcription process.
        device: The device to run the model on (e.g., 'cuda', 'cpu').
        compute_type: The compute type for the model (e.g., 'int8', 'float16').

    Returns:
        The transcribed text as a string.

    Raises:
        TranscriptionError: If the transcription process fails for any reason.

    """
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import Segment

    if device == "cpu" and compute_type == "auto":
        compute_type = "int8"
    try:
        whisper_model = WhisperModel(
            whisper_model_name, device=device, compute_type=compute_type
        )
        logger.info("Initializing transcription")

        segments: Iterable[Segment]
        segments, _ = whisper_model.transcribe(
            str(audio_file_path), beam_size=beam_size
        )
        transcription_text: str = "".join(segment.text for segment in segments)

        logger.info("Transcription completed")
        return transcription_text

    except Exception as e:
        logger.exception("Failed to transcribe audio")
        raise TranscriptionError("Failed to transcribe audio") from e


def fetch_transcription_api(api_url: str, audio_file_path: Path, api_key: str) -> str:
    """Send an audio file to a remote transcription API.

    This function handles the API request and error handling, returning the
    transcription text in memory.

    Args:
        api_url: The URL of the transcription API endpoint.
        audio_file_path: The path to the audio file to be transcribed.
        api_key: The API key for authentication.

    Returns:
        The transcribed text returned by the API as a string.

    Raises:
        TranscriptionError: If the API request fails or returns an error.

    """
    try:
        with audio_file_path.open("rb") as f:
            files: dict[str, IO[bytes]] = {"audio": f}
            logger.info("Initializing transcription")
            response: requests.Response = requests.post(
                api_url,
                files=files,
                timeout=3610,
                headers={"X-Api-Key": api_key},
            )
        response.raise_for_status()
        transcription_text: str = response.json().get("transcription", "")
        logger.info("Transcribed audio successfully")
        return transcription_text

    except requests.exceptions.RequestException as e:
        logger.exception("Failed to transcribe audio")
        raise TranscriptionError("Failed to transcribe audio") from e
    except json.JSONDecodeError as e:
        logger.exception("Failed to parse JSON response")
        raise TranscriptionError("Failed to parse JSON response") from e
