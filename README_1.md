#  🎲 EchoScribes — Turn Your D&D Sessions into Living Stories 🎤 ➔ 📝

A modular Python pipeline to:

- Chunk long multi-speaker audio recordings (one track per speaker),
- Automatically transcribe chunks using WhisperX (via Docker),
- Correct and clean transcription outputs,
- Merge timestamped text chunks into a readable script.

> **Bonus:** Fully extensible for future features like automatic LLM-based summaries.

---

## 📦 Features

- **Audio chunking** with silence detection
- **Automatic WhisperX transcription** (Docker integration)
- **JSON correction and timestamped text formatting**
- **Merging speaker dialogues**
- **Final readable script generation**
- **Low RAM mode available** for lightweight machines
- **Session folder based** — no pollution outside your working directory

---

## 🛠 Installation

First, clone the repository:

```bash
git clone https://github.com/yourusername/audio-transcription-pipeline.git
cd audio-transcription-pipeline
```

Install the package locally:

```bash
pip install -e .
```

Make sure you also have **Docker installed** and a **GPU** available for WhisperX transcription.

---

## 🚀 Usage Example

Inside your session folder (example: `sessionX/` with `tracks/` folder inside):

```python
from audio_pipeline.pipeline import run_pipeline

run_pipeline(
    base_path=".",                  # current working directory
    input_audio_folder="tracks",
    whisperx_output_folder="wx_output",
    corrected_json_folder="json_files",
    srt_folder="srt_files",
    final_folder="final_outputs",
    use_low_ram=True,                # or False
    run_whisperx=True                # or False if you manually transcribe
)
```

**Advanced usage with custom filenames**:

```python
run_pipeline(
    base_path=".",
    input_audio_folder="tracks",
    merged_audio_filename="custom_merged.flac",
    merged_transcript_filename="custom_transcript.srt",
    cleaned_transcript_filename="custom_cleaned.srt",
    final_script_filename="my_final_script.txt",
    run_whisperx=True
)
```

---

## 🔥 Pipeline Overview

| Step | Description |
|:-----|:------------|
| 1 | Chunk long audio files based on detected silences |
| 2 | Transcribe chunks using WhisperX inside Docker |
| 3 | Correct WhisperX JSON outputs |
| 4 | Convert corrected JSONs to timestamped text files |
| 5 | Merge timestamped transcripts chunk-by-chunk |
| 6 | Merge successive speaker entries |
| 7 | Generate final readable script |

---

## 🐳 WhisperX with Docker

The pipeline automatically runs WhisperX using Docker.

Your `Dockerfile` is already included for easy image building:

```bash
docker build -t atp-whisperX-ubuntu22.04-cuda:11.8.0-cudnn8 .
```

The pipeline handles mounting input/output folders and running the container with GPU support.

---

## 📂 Folder Structure Example

```
sessionX/
├── tracks/                     # Your raw audio tracks
├── wx_output/                  # WhisperX output JSONs
├── json_files/                 # Corrected JSONs
├── srt_files/                  # Intermediate timestamped transcripts
└── final_outputs/
    ├── merged_audio.flac       # Full merged audio
    ├── cut_points.txt          # Cut points from silence detection
    ├── merged_transcript.srt   # Merged transcript chunks
    ├── cleaned_transcript.srt  # Cleaned merged transcript
    └── final_script.txt        # Final readable script
```

---

## 🧪 Future work

- **LLM-based automatic summary generation** of final transcripts (🚀 Coming soon)
- **Full WhisperX alternative model integration** (optional)
- **Real-time chunking and processing**

---

## 🧑‍💻 License

This project is open-source and free to use under MIT License.

---
