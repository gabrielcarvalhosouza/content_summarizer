# Content Summarizer

Summarize YouTube videos in seconds using AI (**Google Gemini** or **Ollama**).

**Content-Summarizer** is a CLI tool that downloads YouTube audio, transcribes it locally using **Faster-Whisper** (or uses captions if available), and generates dense, structured summaries using LLMs.

---

## Installation

### Prerequisites

- **Python 3.11+**
- **FFmpeg** (Required for audio processing):
  - **Windows:** `winget install ffmpeg`
  - **Linux:** `sudo apt install ffmpeg`
  - **macOS:** `brew install ffmpeg`

### Installing (Recommended)

Use `uv` or `pipx` to install in an isolated environment:

```bash
# Via UV
uv tool install content-summarizer

# Via Pipx
pipx install content-summarizer
```

---

## First Steps

Before running your first summary, you must configure your default AI provider. You only need to do this once.

### Option A: Using Google Gemini

1. Get your free key at [Google AI Studio](https://aistudio.google.com/).
2. Run:

```bash
content-summarizer config --provider gemini --gemini-key "YOUR_API_KEY_HERE" --gemini-model "3-flash"
```

### Option B: Using Ollama

1. Ensure you have [Ollama](https://ollama.com/) installed and running (`ollama serve`).
2. Run (Example using Mistral, ensure you ran `ollama pull mistral` first):

```bash
content-summarizer config --provider ollama --ollama-model "mistral" --ollama-ctx 16384
```

---

## Usage

The application has two main commands: `summarize` (the main action) and `config` (to set persistent defaults).

### 1. The `summarize` Command

Fetches, processes, and summarizes a video.

**Syntax:**

```bash
content-summarizer summarize "YOUTUBE_URL" [OPTIONS]
```

**Common Options:**

- **Core:**
  - `-o, --output-path <path>`: Directory to save the final `.md` file.
  - `-c, --keep-cache`: Keep temporary files (audio/transcription) after finishing.
  - `--no-terminal`: Do not print the summary to the console.
  - `-q`: Decrease verbosity (warnings only / `-qq` for silent).

- **Audio & Transcription:**
  - `-s, --speed-factor <float>`: Accelerate audio (e.g., `2.0` for 2x speed). Saves transcription time.
  - `-w, --whisper-model <model>`: Model size (`tiny`, `base`, `small`, `medium`, `large-v2`).
  - `--device <device>`: Hardware (`cuda`, `cpu`, `mps`).

- **AI Providers:**
  - `-p, --provider <name>`: `gemini` or `ollama`.
  - `-g, --gemini-model <name>`: E.g., `3-flash`, `3-pro`.
  - `--ollama-model <name>`: E.g., `mistral`, `llama3`.

---

### 2. The `config` Command

Sets default values so you don't have to type flags every time.

**Syntax:**

```bash
content-summarizer config [OPTIONS]
```

**Examples:**

```bash
# Set default download path
content-summarizer config -o "path/to/summaries"

# Set default audio speed to 1.5x
content-summarizer config -s 1.5

# Switch default provider to Ollama
content-summarizer config -p ollama --ollama-model mistral
```

---

## Application Defaults

If you do not provide specific flags or configuration, the application uses the following internal default values:

| Parameter          | Default Value               | Description                                  |
| :----------------- | :-------------------------- | :------------------------------------------- |
| **Provider**       | `gemini`                    | The AI service used for summarization.       |
| **Gemini Model**   | `3-flash`                   | The specific Google Gemini model version.    |
| **Whisper Model**  | `base`                      | The local model size used for transcription. |
| **Speed Factor**   | `1.25`                      | Audio acceleration (1.25x speed).            |
| **Beam Size**      | `5`                         | Beam search size for transcription decoding. |
| **Device**         | `auto`                      | `cuda` (GPU), `mps` (Mac), or `cpu`.         |
| **Compute Type**   | `auto`                      | Quantization (e.g., `int8`, `float16`).      |
| **Ollama URL**     | `http://localhost:11434/v1` | Local address for Ollama server.             |
| **Ollama Context** | `16384`                     | Context window size for local LLMs.          |

---

## Configuration Priority

The application resolves settings in the following order (highest to lowest priority):

1.  **Command-line Flags:** (e.g., passing `-s 2.0` overwrites everything).
2.  **Environment Variables:** Loaded from `.env`.
3.  **User Configuration:** Values set via the `config` command.
4.  **Application Defaults:** Internal hardcoded values.

---

## Important Notes

- **Remote API:** The remote transcription API support (`--api`, `--api-url`) is **DEPRECATED** and will be removed in version 2.0.0. Please migrate to local processing (Whisper).

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
