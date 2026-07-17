from __future__ import annotations

import json
import os
import re
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

SCRIPT_PATH = PROJECT_ROOT / "data" / "podcasts" / "podcast_script.txt"
ANALYSIS_PATH = PROJECT_ROOT / "data" / "analysis" / "analysis.json"
AUDIO_FOLDER = PROJECT_ROOT / "data" / "podcasts" / "audio"
FINAL_AUDIO_PATH = AUDIO_FOLDER / "podcast_episode.wav"

TTS_MODEL = "gpt-4o-mini-tts"

VOICE_SETTINGS = {
    "Alex": {
        "voice": "cedar",
        "instructions": (
            "Speak as Alex, a calm, warm and confident presenter. "
            "Use clear British English pronunciation. Sound natural, engaging and organised."
        ),
    },
    "Jamie": {
        "voice": "marin",
        "instructions": (
            "Speak as Jamie, a friendly, curious and conversational co-host. "
            "Use clear British English pronunciation. Sound interested and natural."
        ),
    },
    "Narrator": {
        "voice": "cedar",
        "instructions": (
            "Speak as a warm, expressive story narrator. Use clear British English "
            "pronunciation, natural dramatic pacing and gentle emotional variation."
        ),
    },
}


@dataclass
class DialogueTurn:
    speaker: str
    text: str


def load_analysis() -> dict[str, Any]:
    if not ANALYSIS_PATH.exists():
        return {"document_mode": "study"}

    try:
        return json.loads(
            ANALYSIS_PATH.read_text(encoding="utf-8", errors="replace")
        )
    except (json.JSONDecodeError, OSError):
        return {"document_mode": "study"}


def clean_dialogue_text(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\[(.*?)\]", r"\1", text)
    text = re.sub(r"^\s*[-•]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_audio_script(script: str, document_mode: str) -> list[DialogueTurn]:
    speaker_pattern = re.compile(
        r"^\s*(?:\*\*)?(Alex|Jamie|Narrator)(?:\*\*)?\s*:\s*(.*)$",
        flags=re.IGNORECASE,
    )

    turns: list[DialogueTurn] = []
    current_speaker: str | None = None
    current_lines: list[str] = []
    unlabelled_story_lines: list[str] = []

    def save_current_turn() -> None:
        nonlocal current_speaker, current_lines

        if not current_speaker:
            return

        text = clean_dialogue_text(" ".join(current_lines))
        if text:
            turns.append(
                DialogueTurn(
                    speaker=current_speaker.capitalize(),
                    text=text,
                )
            )

        current_speaker = None
        current_lines = []

    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = speaker_pattern.match(line)

        if match:
            save_current_turn()
            current_speaker = match.group(1)
            first_text = match.group(2).strip()
            if first_text:
                current_lines.append(first_text)
        elif current_speaker:
            current_lines.append(line)
        elif document_mode == "story":
            if not re.match(r"^(title|chapter|section)\b", line, flags=re.IGNORECASE):
                unlabelled_story_lines.append(line)

    save_current_turn()

    if document_mode == "story" and unlabelled_story_lines and not turns:
        story_text = clean_dialogue_text(" ".join(unlabelled_story_lines))
        if story_text:
            turns.append(DialogueTurn(speaker="Narrator", text=story_text))

    if not turns:
        raise ValueError(
            "No supported spoken content was detected. Study/work/research scripts "
            "need Alex: and Jamie: labels. Story scripts need Narrator: labels or prose."
        )

    return turns


def split_long_text(text: str, character_limit: int = 3600) -> list[str]:
    if len(text) <= character_limit:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        proposed = f"{current_chunk} {sentence}".strip() if current_chunk else sentence

        if len(proposed) <= character_limit:
            current_chunk = proposed
            continue

        if current_chunk:
            chunks.append(current_chunk)

        while len(sentence) > character_limit:
            split_point = sentence.rfind(" ", 0, character_limit)
            if split_point <= 0:
                split_point = character_limit

            chunks.append(sentence[:split_point].strip())
            sentence = sentence[split_point:].strip()

        current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def generate_voice_clip(
    client: OpenAI,
    speaker: str,
    text: str,
    output_path: Path,
) -> None:
    settings = VOICE_SETTINGS[speaker]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=settings["voice"],
        input=text,
        instructions=settings["instructions"],
        response_format="wav",
        speed=1.0,
    ) as response:
        response.stream_to_file(output_path)


def silence_frames(
    wav_parameters: wave._wave_params,
    duration_seconds: float,
) -> bytes:
    frame_count = int(wav_parameters.framerate * duration_seconds)
    bytes_per_frame = wav_parameters.sampwidth * wav_parameters.nchannels
    return b"\x00" * frame_count * bytes_per_frame


def combine_wav_files(
    clip_paths: list[Path],
    output_path: Path,
    pause_seconds: float,
) -> None:
    if not clip_paths:
        raise ValueError("No voice clips were generated.")

    with wave.open(str(clip_paths[0]), "rb") as first_clip:
        expected = first_clip.getparams()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    building_path = output_path.with_name(f"{output_path.stem}_building.wav")

    if building_path.exists():
        building_path.unlink()

    with wave.open(str(building_path), "wb") as output_wav:
        output_wav.setnchannels(expected.nchannels)
        output_wav.setsampwidth(expected.sampwidth)
        output_wav.setframerate(expected.framerate)

        pause = silence_frames(expected, pause_seconds)

        for index, clip_path in enumerate(clip_paths):
            with wave.open(str(clip_path), "rb") as clip:
                current = clip.getparams()

                expected_format = (
                    expected.nchannels,
                    expected.sampwidth,
                    expected.framerate,
                    expected.comptype,
                )
                current_format = (
                    current.nchannels,
                    current.sampwidth,
                    current.framerate,
                    current.comptype,
                )

                if current_format != expected_format:
                    raise ValueError(f"Incompatible WAV settings in {clip_path.name}.")

                output_wav.writeframes(clip.readframes(clip.getnframes()))

            if index < len(clip_paths) - 1:
                output_wav.writeframes(pause)

    try:
        os.replace(building_path, output_path)
    except PermissionError as error:
        raise PermissionError(
            "The existing podcast file is open. Close the audio player or browser tab "
            "and generate it again."
        ) from error


def main() -> None:
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: OPENAI_API_KEY was not found.")
        return

    if not SCRIPT_PATH.exists():
        print("ERROR: Podcast script was not found.")
        print("Run backend/podcast_generator.py first.")
        return

    script = SCRIPT_PATH.read_text(encoding="utf-8", errors="replace").strip()
    if not script:
        print("ERROR: The podcast script is empty.")
        return

    try:
        analysis = load_analysis()
        document_mode = str(analysis.get("document_mode", "study")).lower()
        turns = parse_audio_script(script, document_mode)

        print(f"Detected {len(turns)} spoken turns for {document_mode} mode.")

        client = OpenAI(api_key=api_key)

        with tempfile.TemporaryDirectory(
            prefix="podcastai_audio_",
            ignore_cleanup_errors=True,
        ) as temporary_directory:
            temp_folder = Path(temporary_directory)
            clip_paths: list[Path] = []
            clip_number = 1

            for turn_number, turn in enumerate(turns, start=1):
                chunks = split_long_text(turn.text)

                for chunk_number, chunk in enumerate(chunks, start=1):
                    print(f"Recording {turn_number}/{len(turns)} - {turn.speaker}")

                    clip_path = temp_folder / (
                        f"{clip_number:04d}_{turn.speaker.lower()}_{chunk_number}.wav"
                    )

                    generate_voice_clip(
                        client,
                        turn.speaker,
                        chunk,
                        clip_path,
                    )

                    clip_paths.append(clip_path)
                    clip_number += 1

            pause_seconds = 0.70 if document_mode == "story" else 0.55

            print("\nCombining generated audio...")
            combine_wav_files(
                clip_paths,
                FINAL_AUDIO_PATH,
                pause_seconds,
            )

        print("\nSUCCESS: Audio created!")
        print(f"Saved to: {FINAL_AUDIO_PATH}")
        print("Disclosure: This audio uses AI-generated voices.")

    except Exception as error:
        print(f"\nERROR: Podcast audio generation failed: {error}")


if __name__ == "__main__":
    main()
