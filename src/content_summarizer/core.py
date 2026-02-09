"""Orchestrates the main application logic and configuration building.

This module acts as the central hub of the application. It contains the primary
pipeline for summarizing content, the logic for handling user configurations,
and the functions responsible for building the main application state object.

Classes:
    SetupError: Custom exception for errors during the initial setup phase.
    PipelineError: Custom exception for errors during the main processing pipeline.
    AppConfig: Dataclass holding all shared application configurations and services.

Functions:
    build_app_config: Initializes and returns the main AppConfig object.
    summarize_video_pipeline: Runs the complete video summarization workflow.
    handle_config_command: Processes and saves user configuration settings.
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
import locale
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import Any

from dotenv import load_dotenv

from content_summarizer.data.data_models import VideoMetadata
from content_summarizer.managers.cache_manager import CacheManager
from content_summarizer.managers.config_manager import ConfigManager
from content_summarizer.managers.path_manager import PathManager
from content_summarizer.processors.audio_processor import AudioProcessor
from content_summarizer.services.summary_service import generate_summary
from content_summarizer.services.transcription_service import (
    fetch_transcription_api,
    fetch_transcription_local,
)
from content_summarizer.services.youtube_service import YoutubeService


class SetupError(Exception):
    """Custom exception for errors during the setup process."""

    pass


class PipelineError(Exception):
    """Custom exception for errors during the pipeline execution."""

    pass


@dataclass
class AppConfig:
    """Holds all shared application configurations and service instances.

    This dataclass acts as a dependency injection container, making it easy to
    pass all necessary services and settings throughout the application.

    Attributes:
        logger: The configured logger instance for the application.
        path_manager: The manager for all application paths.
        youtube_service: The service for interacting with YouTube.
        cache_manager: The manager for cache file operations.
        config_manager: The manager for user configuration files.
        url: The URL of the content to be summarized.
        output_path: The root directory for output files.
        keep_cache: A boolean to prevent cache deletion.
        quiet: The console verbosity level.
        speed_factor: The audio acceleration factor.
        gemini_key: The API key for the Gemini service.
        gemini_model_name: The name of the Gemini model being used.
        ollama_model_name: The name of the Ollama model being used.
        ollama_url: The URL for the Ollama service.
        ollama_ctx: The context window for Ollama.
        api: A boolean to select the remote transcription API.
        api_url: The URL for the remote transcription API.
        api_key: The API key for the remote transcription API.
        whisper_model: The name of the local Whisper model being used.
        beam_size: The beam size for local transcription.
        device: The device for local transcription (e.g., 'cuda', 'cpu').
        no_terminal: A boolean to disable terminal output of the summary.
        user_language: The detected user system language code.

    """

    logger: logging.Logger
    path_manager: PathManager
    youtube_service: YoutubeService
    cache_manager: CacheManager
    config_manager: ConfigManager
    url: str
    output_path: Path | None
    keep_cache: bool
    quiet: int
    speed_factor: float
    provider: str
    gemini_key: str | None
    gemini_model_name: str
    ollama_model_name: str
    ollama_url: str
    ollama_ctx: int
    api: bool
    api_url: str | None
    api_key: str | None
    whisper_model: str
    beam_size: int
    device: str
    compute_type: str
    no_terminal: bool
    user_language: str


def _resolve_config(
    args: argparse.Namespace, path_manager: PathManager, config_manager: ConfigManager
) -> dict[str, Any]:
    """Resolve the final configuration from multiple sources.

    This function builds the final configuration dictionary by layering sources
    in a specific order of precedence:
    1. Default hardcoded values.
    2. User's saved `config.json` file.
    3. Environment variables from a `.env` file.
    4. Command-line arguments.

    Args:
        args: The parsed command-line arguments.
        path_manager: The application's path manager.
        config_manager: The manager for the user's configuration file.

    Returns:
        A dictionary containing the final, resolved configuration.

    """
    final_config: dict[str, Any] = {
        "output_path": None,
        "keep_cache": False,
        "quiet": 0,
        "speed_factor": 1.25,
        "provider": "gemini",
        "gemini_key": "",
        "gemini_model_name": "3-flash",
        "ollama_model_name": "",
        "ollama_url": "http://localhost:11434/v1",
        "ollama_ctx": 16384,
        "api": False,
        "api_url": "",
        "api_key": "",
        "whisper_model": "base",
        "beam_size": 5,
        "device": "auto",
        "compute_type": "auto",
        "no_terminal": False,
    }

    user_saved_config: dict[str, Any] = config_manager.load_config()

    final_config.update(user_saved_config)

    load_dotenv(path_manager.parent_path / ".env")
    gemini_key: str | None = os.getenv("GEMINI_API_KEY")
    ollama_url: str | None = os.getenv("OLLAMA_API_URL")
    api_url: str | None = os.getenv("API_URL")
    api_key: str | None = os.getenv("TRANSCRIPTION_API_KEY")

    if gemini_key:
        final_config["gemini_key"] = gemini_key
    if ollama_url:
        final_config["ollama_url"] = ollama_url
    if api_url:
        final_config["api_url"] = api_url
    if api_key:
        final_config["api_key"] = api_key

    dict_args = vars(args)
    for key, value in dict_args.items():
        if value is None:
            continue
        if key == "command":
            continue
        final_config[key] = value

    return final_config


def _check_required_config_params(
    final_config: dict[str, Any], logger: logging.Logger
) -> None:
    """Validate that all required configuration parameters are present.

    Checks the resolved configuration dictionary for essential values, raising
    an error if a required parameter is missing.

    Args:
        final_config: The resolved configuration dictionary.
        logger: The application's logger.

    Raises:
        ValueError: If a required parameter is missing.

    """
    if final_config.get("provider") == "gemini" and not final_config.get("gemini_key"):
        logger.error("Gemini API key is required, use the --gemini-key flag")
        raise ValueError("Gemini API key is required")

    if final_config.get("provider") == "ollama" and not final_config.get(
        "ollama_model_name"
    ):
        logger.error("Ollama model name is required, use the --ollama-model-name flag")
        raise ValueError("Ollama model name is required")

    if not final_config.get("api"):
        return

    if not final_config.get("api_url"):
        logger.error("API URL is required when API mode is enabled")
        raise ValueError("API URL is required when API mode is enabled")

    if not final_config.get("api_key"):
        logger.error("API key is required when API mode is enabled")
        raise ValueError("API key is required when API mode is enabled")


def _get_user_system_language(logger: logging.Logger) -> str:
    """Detect and normalize the user's system language.

    Tries to get the system's locale and formats it into a web-compatible
    language code (e.g., 'en-US'). Defaults to 'en-US' if detection fails.

    Args:
        logger: The application's logger.

    Returns:
        The normalized language code string.

    """
    DEFAULT_LOCALE: str = "en_US"
    lang_code: str | None

    try:
        locale.setlocale(locale.LC_ALL, "")
        lang_code, _ = locale.getlocale()
    except locale.Error:
        lang_code = None

    if not lang_code or lang_code == "C":
        env_vars: list[str] = ["LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"]

        for var in env_vars:
            value = os.environ.get(var)
            if value and value not in ("C", "C.UTF-8", "POSIX"):
                lang_code = value
                break

    if not lang_code:
        logger.warning("Failed to detect locale, using default: %s", DEFAULT_LOCALE)
        lang_code = DEFAULT_LOCALE

    lang_code = lang_code.split(".")[0].replace("_", "-")
    logger.info("Detected locale: %s", lang_code)
    return lang_code


def _build_app_config(
    args: argparse.Namespace, logger: logging.Logger, path_manager: PathManager
) -> AppConfig:
    """Initialize and build the complete AppConfig object.

    This function orchestrates the entire setup process: it resolves the
    configuration from all sources, initializes all service and manager
    classes, and assembles them into a single AppConfig instance.

    Args:
        args: The parsed command-line arguments.
        logger: The application's logger.
        path_manager: The application's path manager.

    Returns:
        A populated AppConfig instance with all dependencies.

    """
    config_manager: ConfigManager = ConfigManager(path_manager.config_file_path)
    youtube_service: YoutubeService = YoutubeService()
    cache_manager: CacheManager = CacheManager()

    final_config: dict[str, Any] = _resolve_config(args, path_manager, config_manager)

    _check_required_config_params(final_config, logger)

    user_language: str = _get_user_system_language(logger)

    return AppConfig(
        logger=logger,
        path_manager=path_manager,
        youtube_service=youtube_service,
        cache_manager=cache_manager,
        config_manager=config_manager,
        url=final_config["url"],
        output_path=(
            Path(final_config["output_path"]) if final_config["output_path"] else None
        ),
        keep_cache=final_config["keep_cache"],
        quiet=final_config["quiet"],
        speed_factor=final_config["speed_factor"],
        provider=final_config["provider"],
        gemini_key=final_config["gemini_key"],
        gemini_model_name=final_config["gemini_model_name"],
        ollama_model_name=final_config["ollama_model_name"],
        ollama_url=final_config["ollama_url"],
        ollama_ctx=final_config["ollama_ctx"],
        api=final_config["api"],
        api_url=final_config["api_url"],
        api_key=final_config["api_key"],
        whisper_model=final_config["whisper_model"],
        beam_size=final_config["beam_size"],
        user_language=user_language,
        no_terminal=final_config["no_terminal"],
        device=final_config["device"],
        compute_type=final_config["compute_type"],
    )


def _save_caption(config: AppConfig, caption: str, log_success: bool) -> None:
    """Save the provided caption text to a cache file."""
    config.cache_manager.save_text_file(
        caption, config.path_manager.caption_file_path, log_success
    )


def _save_accelerated_audio(config: AppConfig, accelerated_audio_path: Path) -> None:
    """Ensure the accelerated audio file exists, creating it if necessary.

    This function orchestrates the audio processing. It first ensures the
    original audio is downloaded and then accelerates it to the target speed
    if the accelerated version doesn't already exist in the cache.
    """
    audio_processor: AudioProcessor = AudioProcessor(
        config.path_manager.audio_file_path, accelerated_audio_path
    )

    if not config.path_manager.audio_file_path.exists():
        config.youtube_service.audio_download(config.path_manager.audio_file_path)

    if not accelerated_audio_path.exists():
        audio_processor.accelerate_audio(config.speed_factor)


def _save_transcription(
    config: AppConfig,
    accelerated_audio_path: Path,
    transcription_file_path: Path,
    log_success: bool,
) -> None:
    """Ensure the transcription file exists, creating it if necessary.

    This function orchestrates the transcription process. It selects the
    appropriate transcription method (local or API) based on the user's
    configuration and saves the resulting text to a cache file if it
    doesn't already exist.
    """
    if not transcription_file_path.exists():

        def _fetch_from_api() -> str:
            """Handle API transcription, including prerequisite checks."""
            assert config.api_url, "API URL is required for API mode"
            assert config.api_key, "API key is required for API mode"
            return fetch_transcription_api(
                config.api_url,
                accelerated_audio_path,
                config.api_key,
            )

        transcription_fetcher: dict[bool, Callable[[], str]] = {
            True: _fetch_from_api,
            False: lambda: fetch_transcription_local(
                accelerated_audio_path,
                config.whisper_model,
                config.beam_size,
                config.device,
                config.compute_type,
            ),
        }

        selected_fetcher: Callable[[], str] = transcription_fetcher[config.api]
        transcription: str = selected_fetcher()

        if not transcription:
            raise PipelineError("Failed to fetch transcription")

        config.cache_manager.save_text_file(
            transcription, transcription_file_path, log_success
        )


def _handle_metadata(config: AppConfig, log_success: bool) -> None:
    """Manage the creation and state of the video's metadata file.

    This function ensures the metadata file exists and correctly handles the
    persistence of the 'keep_cache' flag. It reads any existing metadata to
    check if the cache was previously marked for persistence. The flag is then
    made "sticky," meaning once it's set to True, it will not be reverted to
    False by subsequent runs that don't use the --keep-cache flag.

    Args:
        config: The application's configuration object, containing all necessary
                services and settings.
        log_success: Whether to log a success message upon saving the file.

    """
    _existing_keep_cache: bool = config.cache_manager.read_keep_cache_flag(
        config.path_manager.metadata_file_path
    )
    _final_keep_cache: bool = _existing_keep_cache or config.keep_cache
    video_metadata: VideoMetadata = VideoMetadata(
        id=config.youtube_service.video_id,
        url=config.url,
        title=config.youtube_service.title,
        author=config.youtube_service.author,
        keep_cache=_final_keep_cache,
    )

    config.cache_manager.save_metadata_file(
        video_metadata,
        config.path_manager.metadata_file_path,
        log_success,
    )


def _prepare_source_file(
    config: AppConfig, caption: str | None, log_success: bool
) -> Path:
    """Prepare the source text file for summarization.

    This function acts as a dispatcher. If a manual caption is available,
    it saves it to a file. Otherwise, it triggers the full audio download
    and transcription pipeline to generate the source file.

    Args:
        config: The application's configuration object.
        caption: The pre-fetched caption text, or None.
        log_success: Whether to log a success message.

    Returns:
        The path to the prepared source text file (caption or transcription).

    """
    if caption:
        _save_caption(config, caption, log_success)
        return config.path_manager.caption_file_path

    accelerated_audio_path: Path = config.path_manager.get_accelerated_audio_path(
        config.speed_factor
    )
    transcription_file_path: Path = config.path_manager.get_transcription_path(
        config.whisper_model, config.speed_factor, config.beam_size
    )
    _save_accelerated_audio(config, accelerated_audio_path)
    _save_transcription(
        config, accelerated_audio_path, transcription_file_path, log_success
    )

    return transcription_file_path


def _prepare_video_pipeline(logger: logging.Logger, config: AppConfig) -> None:
    """Prepare the application state for processing.

    This function loads the video information from the URL provided in the
    configuration and sets the video ID in the path manager. This is the
    initialization phase of the pipeline.

    Args:
        logger: The application's configured logger.
        config: The application's configuration object.

    Raises:
        SetupError: If the video cannot be loaded or the ID cannot be set.

    """
    try:
        config.youtube_service.load_from_url(config.url)
        config.path_manager.set_video_id(config.youtube_service.video_id)
    except Exception as e:
        logger.exception("An error occurred during the setup")
        raise SetupError("An error occurred during the setup") from e


def _summarize_video_pipeline(
    logger: logging.Logger,
    config: AppConfig,
) -> None:
    """Execute the core summarization logic.

    This function runs the main stages of the pipeline: finding captions,
    downloading/transcribing audio (if necessary), generating the summary
    using AI, and saving/displaying the results. It also handles cache
    cleanup upon completion.

    Args:
        logger: The application's configured logger.
        config: The application's configuration object.

    Raises:
        PipelineError: If any error occurs during the processing stages.

    """
    try:
        _log_success: bool = True
        if not config.keep_cache:
            _log_success = False

        caption: str | None = config.youtube_service.find_best_captions(
            config.user_language
        )
        _handle_metadata(config, _log_success)

        source_path = _prepare_source_file(config, caption, _log_success)

        summary_file_path: Path = config.path_manager.get_summary_path(
            config.gemini_model_name,
            config.user_language,
            config.whisper_model,
            config.speed_factor,
            config.beam_size,
        )

        summary: str | None = None
        if summary_file_path.exists():
            config.logger.info("Summary found in cache, loading from file")
            with summary_file_path.open("r", encoding="utf-8") as f:
                summary = f.read()

        if not summary:
            summary = generate_summary(
                config,
                source_path,
            )

        if summary:
            config.cache_manager.save_text_file(
                summary, summary_file_path, _log_success
            )

        if summary and not config.no_terminal:
            from rich.console import Console
            from rich.markdown import Markdown

            console: Console = Console()
            markdown_summary: Markdown = Markdown(summary)
            console.print("-" * console.width)
            console.print(markdown_summary)
            console.print("-" * console.width)

        if summary and config.output_path:
            summary_output_path: Path = config.path_manager.get_final_summary_path(
                config.youtube_service.title, config.output_path
            )
            config.cache_manager.save_text_file(
                summary, summary_output_path, log_success=False
            )
            logger.info(f"Summary saved to {summary_output_path}")

    except Exception as e:
        config.logger.exception("An error occurred during the pipeline")
        raise PipelineError("An error occurred during the pipeline:") from e
    finally:
        _keep_cache: bool = config.cache_manager.read_keep_cache_flag(
            config.path_manager.metadata_file_path
        )
        if config.path_manager.video_dir_path.exists() and not _keep_cache:
            rmtree(config.path_manager.video_dir_path)
            logger.info("Cache cleared")


def handle_summarize_command(
    args: argparse.Namespace, logger: logging.Logger, path_manager: PathManager
) -> None:
    """Run the complete video summarization workflow.

    This function acts as the controller for the 'summarize' command. It
    builds the configuration, orchestrates the setup phase, and triggers
    the execution phase.

    Args:
        args: The parsed command-line arguments.
        logger: The application's configured logger.
        path_manager: The application's path manager.

    """
    if args.api or args.api_url:
        logger.warning(
            "DEPRECATION WARNING: The remote API feature (--api, --api-url)"
            " is deprecated and WILL BE REMOVED in version 2.0.0. "
        )

    config: AppConfig = _build_app_config(args, logger, path_manager)

    _prepare_video_pipeline(logger, config)

    _summarize_video_pipeline(logger, config)


def handle_config_command(
    args: argparse.Namespace, logger: logging.Logger, path_manager: PathManager
) -> None:
    """Process and save user configuration settings.

    Reads the current configuration, updates it with any new values provided
    by the user via command-line arguments, and saves it back to the
    `config.json` file.

    Args:
        args: The parsed command-line arguments from the user.
        logger: The application's configured logger.
        path_manager: The application's path manager.

    Raises:
        OSError: If the configuration file cannot be written.

    """
    config_manager: ConfigManager = ConfigManager(path_manager.config_file_path)

    try:
        logger.info("Saving configuration")
        current_configs: dict[str, Any] = config_manager.load_config(is_config=True)
        dict_args: dict[str, Any] = vars(args)

        for key, value in dict_args.items():
            if value is None:
                continue
            if key == "command":
                continue
            if key == "output-path":
                current_configs[key] = str(value)
                continue
            current_configs[key] = value

        config_manager.save_config(current_configs)
        logger.info("Configuration saved successfully")
    except OSError:
        logger.exception("Failed to save configuration")
        raise
