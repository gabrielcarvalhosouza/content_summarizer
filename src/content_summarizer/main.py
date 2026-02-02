"""Main entry point for the Content Summarizer application.

This module is responsible for initializing the application, parsing command-line
arguments, setting up logging, and dispatching tasks to the appropriate
core functions. It acts as the primary orchestrator and final error handler.

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
import sys

from content_summarizer.cli import parse_arguments
from content_summarizer.core import handle_config_command, summarize_video_pipeline
from content_summarizer.managers.path_manager import PathManager
from content_summarizer.utils.logger_config import setup_logging
from content_summarizer.utils.warning_config import setup_warnings


def main() -> None:
    """Run the main application logic.

    This function initializes all necessary components (warnings, args, logging),
    acts as a dispatcher to call the correct core function based on the
    user's command, and serves as the final safety net, catching any
    unhandled exceptions.

    """
    setup_warnings()
    args = parse_arguments()
    path_manager: PathManager = PathManager()

    quiet_level = getattr(args, "quiet", 0)

    setup_logging(path_manager.log_file_path, quiet_level)
    logger: logging.Logger = logging.getLogger(__name__)
    try:
        if args.command == "config":
            handle_config_command(args, logger, path_manager)
            return
        summarize_video_pipeline(args, logger, path_manager)
        logger.info("Application completed successfully")
    except Exception:
        logger.info(f"Full traceback saved to: {path_manager.log_file_path}")
        logger.critical("Fatal error occurred. Exiting application")

        sys.exit(1)


if __name__ == "__main__":
    main()
