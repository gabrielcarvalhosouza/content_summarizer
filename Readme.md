# Content Summarizer

## A tool to summarize YouTube videos using AI

Content-Summarizer is a simple CLI program that summarize Youtube videos using Google Gemini.

## Installation

### Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.11+**
2.  **FFmpeg** (Required for audio processing):
    - **Windows:** `winget install ffmpeg` (or add to PATH manually)
    - **Linux:** `sudo apt install ffmpeg`
    - **macOS:** `brew install ffmpeg`
3.  **Ollama** (Optional - Only if you want to run local models):
    - Download from [ollama.com](https://ollama.com)
    - Pull a model: `ollama pull mistral` (or llama3, gemma, etc.)

### Recommended Installation

The best way to install is using `uv` to keep the environment isolated:

```bash
uv tool install content-summarizer
```

Or if you prefer to use `pipx`:

```bash
pipx install content-summarizer
```

You can also use the standard `pip`, but be aware that it will install the package in your global or current environment:

```bash
pip install content-summarizer
```

## Usage

The application has two main commands: `summarize` and `config`.

- `summarize`: Fetches and summarizes a given YouTube URL. This is the main command.
- `config`: Sets default values for flags, so you don't have to type them on every run. These settings are saved in a system-specific user configuration directory.

For a full list of all commands and flags, run `content-summarizer --help`.

### The `summarize` Command

**Basic Usage**

```bash
content-summarizer summarize "YOUR_YOUTUBE_URL_HERE"
```

#### Common Summarize Flags

```bash
# Change the audio speed factor for faster transcriptions
content-summarizer summarize "YOUR_YOUTUBE_URL_HERE" -s 2.5

# Change Whisper (Faster-Whisper) Model
content-summarizer summarize "YOUR_YOUTUBE_URL_HERE" -w large-v2

# Change Gemini Model
content-summarizer summarize "YOUR_YOUTUBE_URL_HERE" -g 2.5-pro

# Decrease console verbosity (shows only warnings and errors)
content-summarizer summarize "YOUR_YOUTUBE_URL_HERE" -q

# Make the console output completely silent (summary output still works)
content-summarizer summarize "YOUR_YOUTUBE_URL_HERE" -qq

# Disable the summary output in the terminal
content-summarizer summarize "YOUR_YOUTUBE_URL_HERE" --no-terminal

# Specify an output path for the creation of an output summary file alongside the normal terminal output
content-summarizer summarize "YOUR_YOUTUBE_URL_HERE" -o "YOUR_OUTPUT_PATH_HERE"

# Keep the cache directory after execution for re-runs
content-summarizer summarize "YOUR_YOUTUBE_URL_HERE" -c
```

### The `config` Command

#### Common Config Flags

```bash
# Specify default output path
content-summarizer config -o "YOUR_OUTPUT_PATH_HERE"

# Specify default speed factor
content-summarizer config -s 1.5

# Specify default API KEY
content-summarizer config --api-key "YOUR_OWN_DEPLOYED_API_KEY_HERE"

# Specify default Google AI Studio API KEY
content-summarizer config --gemini-key "YOUR_GOOGLE_AI_KEY_HERE"
```

## Configuration

The application resolves settings with the following priority order:

1. Command-line Flags: Always takes top priority for the current run.

2. Environment Variables: Loaded from an `.env` file or system environments.

3. User Configuration: Defaults set via the `config` command.

4. **Application Defaults:** The program's default values. You can see them [right here](https://github.com/gabrielcarvalhosouza/content_sumarizer/blob/9f8329ff23bd8e070ad6cfd3770724981ea9d7ce/src/core.py#L148-L162).

## Using the Remote Transcription API

The `--api` flag allows you to offload transcription to a remote server. This project includes a simple Flask API in the `flask_api/` directory that you can deploy yourself.

To set up the API, you will need to install its specific dependencies:

```bash
# Navigate to the api directory
cd flask_api/

# Install the API dependencies
uv pip install -r requirements.txt
```

To use this feature, you must:

1.  Deploy the application found in the `flask_api/` folder to a server of your choice.
2.  Use the `--api-url` and `--api-key` flags (or set them as defaults using the `config` command) to point the CLI to your deployed API.

Fell free to use your own API key, you just need to put it in the API `.env`. I recommend the use of the following script to generate a safe API key, just copy and paste in any terminal:

```bash

python -c "import secrets, string; print(''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(30)))"
```

A detailed deployment guide is beyond the scope of this README.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
