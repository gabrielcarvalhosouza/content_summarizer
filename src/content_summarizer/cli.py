"""Defines the command-line interface for the application.

This module uses the argparse library to build the entire CLI, including
commands, sub-commands, and all their respective arguments and options.
It is the main entry point for user interaction.

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

import argparse
from importlib import metadata
from pathlib import Path

WHISPER_MODEL_LIST = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
]

GEMINI_MODEL_LIST = [
    "2.5-flash",
    "2.5-pro",
    "3-flash",
    "3-pro",
]

DEVICES_LIST = [
    "cuda",
    "mps",
    "cpu",
    "auto",
]

COMPUTE_TYPES_LIST = [
    "int8",
    "float16",
    "float32",
    "int8_float16",
    "auto",
]

PROVIDERS_LIST = ["gemini", "ollama"]


def parse_arguments() -> argparse.Namespace:
    """Set up and parse all command-line arguments.

    Builds the complete CLI structure, defining the main parser,
    the 'summarize' and 'config' subparsers, and all their options.

    Returns:
        An object containing the parsed command-line arguments.

    """
    parser = argparse.ArgumentParser(
        prog="content-summarizer",
        description=(
            "An automated tool for transcribing and summarizing YouTube videos "
            "using local Whisper models and AI providers like Gemini and Ollama."
        ),
        epilog=(
            "Quick Start:\n"
            "  1. Configure your API key: content-summarizer config "
            "--gemini-key YOUR_KEY\n"
            "  2. Summarize a video: content-summarizer summarize "
            "https://www.youtube.com/watch?v=VIDEO_ID\n"
            "\n"
            "For more details on a specific command, run: "
            "content-summarizer <command> --help"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    try:
        version = metadata.version("content-summarizer")
    except metadata.PackageNotFoundError:
        version = "0.0.0 (local development)"

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {version}",
        help="Show the application's version and exit.",
    )

    subparsers = parser.add_subparsers(dest="command")

    parser_summarize = subparsers.add_parser(
        "summarize",
        help="Summarize a YouTube video from a given URL.",
    )

    core_group = parser_summarize.add_argument_group("Core Options")
    core_group.add_argument(
        "url", type=str, help="The URL of the YouTube video to summarize."
    )
    core_group.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Custom directory for saving generated files.",
    )
    core_group.add_argument(
        "-c",
        "--keep-cache",
        action="store_true",
        help="Keep downloaded and processed files after completion.",
    )
    core_group.add_argument(
        "--no-terminal",
        action="store_true",
        help="Do not print the final summary to the console.",
    )
    core_group.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="Lower verbosity level (-q for warnings, -qq for silent).",
    )

    audio_group = parser_summarize.add_argument_group("Audio Options")
    audio_group.add_argument(
        "-s",
        "--speed-factor",
        type=float,
        help="Factor to speed up audio processing (e.g., 2.0 for 2x speed).",
    )

    whisper_group = parser_summarize.add_argument_group(
        "Local Transcription Options (Whisper)"
    )
    whisper_group.add_argument(
        "-w",
        "--whisper-model",
        type=str,
        choices=WHISPER_MODEL_LIST,
        help="Whisper model size for local transcription.",
    )
    whisper_group.add_argument(
        "-b",
        "--beam-size",
        type=int,
        help="Beam size for the transcription decoding process.",
    )
    whisper_group.add_argument(
        "--device",
        type=str,
        choices=DEVICES_LIST,
        help="Hardware device for processing (e.g., 'cuda', 'cpu').",
    )
    whisper_group.add_argument(
        "--compute-type",
        type=str,
        choices=COMPUTE_TYPES_LIST,
        help="Quantization type for local computation.",
    )

    api_group = parser_summarize.add_argument_group(
        "API Transcription Options (Deprecated)"
    )
    api_group.add_argument(
        "-a",
        "--api",
        action="store_true",
        help="[Deprecated] Use a remote API for transcription.",
    )
    api_group.add_argument(
        "--api-url",
        type=str,
        help="[Deprecated] URL of the remote transcription service.",
    )
    api_group.add_argument(
        "--api-key",
        type=str,
        help="[Deprecated] Authentication key for the remote transcription service.",
    )

    provider_group = parser_summarize.add_argument_group("Summarization Options")
    provider_group.add_argument(
        "-p",
        "--provider",
        type=str,
        choices=PROVIDERS_LIST,
        help="AI service provider for content summarization.",
    )

    gemini_group = parser_summarize.add_argument_group("Gemini AI Settings")
    gemini_group.add_argument(
        "--gemini-key",
        type=str,
        help="API Key for Google Gemini AI.",
    )
    gemini_group.add_argument(
        "-g",
        "--gemini-model",
        type=str,
        dest="gemini_model_name",
        choices=GEMINI_MODEL_LIST,
        help="Specific Gemini model version to use.",
    )

    ollama_group = parser_summarize.add_argument_group("Ollama AI Settings")
    ollama_group.add_argument(
        "--ollama-model",
        dest="ollama_model_name",
        type=str,
        help="Name of the model running on Ollama.",
    )
    ollama_group.add_argument(
        "--ollama-url",
        type=str,
        help="Endpoint URL for the Ollama instance.",
    )
    ollama_group.add_argument(
        "--ollama-ctx",
        type=int,
        help="Context window size for the Ollama model.",
    )

    parser_config = subparsers.add_parser(
        "config",
        help="Manage default configuration settings for the application.",
    )

    core_cfg = parser_config.add_argument_group("Core Settings")
    core_cfg.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Default directory for saving generated files.",
    )

    audio_cfg = parser_config.add_argument_group("Audio Settings")
    audio_cfg.add_argument(
        "-s",
        "--speed-factor",
        type=float,
        help="Default factor to speed up audio processing.",
    )

    whisper_cfg = parser_config.add_argument_group(
        "Local Transcription Settings (Whisper)"
    )
    whisper_cfg.add_argument(
        "-w",
        "--whisper-model",
        type=str,
        choices=WHISPER_MODEL_LIST,
        help="Default Whisper model size for local transcription.",
    )
    whisper_cfg.add_argument(
        "-b",
        "--beam-size",
        type=int,
        help="Default beam size for transcription.",
    )
    whisper_cfg.add_argument(
        "--device",
        type=str,
        choices=DEVICES_LIST,
        help="Default hardware device (e.g., 'cuda', 'cpu').",
    )
    whisper_cfg.add_argument(
        "--compute-type",
        type=str,
        choices=COMPUTE_TYPES_LIST,
        help="Default quantization type for computation.",
    )

    api_cfg = parser_config.add_argument_group(
        "API Transcription Settings (Deprecated)"
    )
    api_cfg.add_argument(
        "--api-url",
        type=str,
        help="[Deprecated] Default URL for remote transcription.",
    )
    api_cfg.add_argument(
        "--api-key",
        type=str,
        help="[Deprecated] Default key for remote transcription.",
    )

    provider_cfg = parser_config.add_argument_group("Summarization Settings")
    provider_cfg.add_argument(
        "-p",
        "--provider",
        type=str,
        choices=PROVIDERS_LIST,
        help="Default AI service provider for summarization.",
    )

    gemini_cfg = parser_config.add_argument_group("Gemini AI Settings")
    gemini_cfg.add_argument(
        "--gemini-key",
        type=str,
        help="Default API Key for Google Gemini AI.",
    )
    gemini_cfg.add_argument(
        "-g",
        "--gemini-model",
        type=str,
        dest="gemini_model_name",
        choices=GEMINI_MODEL_LIST,
        help="Default Gemini model version.",
    )

    ollama_cfg = parser_config.add_argument_group("Ollama AI Settings")
    ollama_cfg.add_argument(
        "--ollama-model",
        dest="ollama_model_name",
        type=str,
        help="Default model name for Ollama.",
    )
    ollama_cfg.add_argument(
        "--ollama-url",
        type=str,
        help="Default endpoint URL for Ollama.",
    )
    ollama_cfg.add_argument(
        "--ollama-ctx",
        type=int,
        help="Default context window size for Ollama.",
    )

    return parser.parse_args()
