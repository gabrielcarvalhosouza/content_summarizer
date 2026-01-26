"""Flask API for transcribing audio files using a local Whisper model.

This module provides a single '/transcribe' endpoint that accepts POST
requests with an audio file. It handles API key authentication, rate limiting,
and orchestrates the transcription process, returning the result as JSON.

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

import hmac
import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import dotenv
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from flask import Flask, Response, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.datastructures import FileStorage

app: Flask = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",
)


parent_path: Path = Path(__file__).parent
logfile_path: Path = parent_path / "app.log"
log_formatter: logging.Formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"
)
file_handler: logging.FileHandler = logging.FileHandler(
    logfile_path, mode="a", encoding="utf-8"
)
file_handler.setFormatter(log_formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

logger.addHandler(file_handler)


dotenv.load_dotenv(parent_path / ".env")
api_secret_key: str | None = os.getenv("API_SECRET_KEY")

if api_secret_key is None:
    logger.error("API_SECRET_KEY environment variable not set")
    raise ValueError("API_SECRET_KEY environment variable not set")

whisper_model = WhisperModel("base", device="cpu", compute_type="int8")


@app.route("/transcribe", methods=["POST"])
@limiter.limit("2 per minute, 5 per day")
def transcribe() -> Response | tuple[Response, int]:
    """Handle audio transcription requests.

    Accepts a POST request with a multipart form containing an 'audio' file.
    It requires a valid 'X-Api-Key' header for authentication.

    Returns:
        - 200 OK: A JSON object with the transcription text.
        - 400 Bad Request: If no audio file is provided.
        - 401 Unauthorized: If the API key is missing or invalid.
        - 429 Too Many Requests: If the rate limit is exceeded.
        - 500 Internal Server Error: If an unexpected error occurs.

    """
    logger.warning(
        "DEPRECATION WARNING: The remote API feature (--api, --api-url)"
        " is deprecated and WILL BE REMOVED in version 2.0.0. "
    )

    provided_api_key: str | None = request.headers.get("X-Api-Key")
    assert api_secret_key
    if not provided_api_key or not hmac.compare_digest(
        provided_api_key, api_secret_key
    ):
        logger.warning("Unauthorized request from %s", request.remote_addr)
        return jsonify({"error": "Unauthorized"}), 401

    logger.info("Request received from %s", request.remote_addr)
    audio_file: FileStorage | None = request.files.get("audio")
    if not audio_file:
        logger.warning("No audio file uploaded")
        return jsonify({"error": "No audio file uploaded"}), 400

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "audio.mp3"

            audio_file.save(temp_path)

            logger.info("Initializing transcription")

            segments: Iterable[Segment]
            segments, _ = whisper_model.transcribe(str(temp_path), beam_size=5)
            transcription_text: str = "".join(segment.text for segment in segments)

            logger.info("Transcription completed")
            return jsonify({"transcription": transcription_text})

    except Exception:
        logger.exception("Error occurred during transcription")
        return jsonify(
            {"error": "An internal error occurred during transcription."}
        ), 500


if __name__ == "__main__":
    app.run(debug=True, port=8000)
