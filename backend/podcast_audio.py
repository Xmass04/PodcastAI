import os
import re
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

SCRIPT_PATH = (
    PROJECT_ROOT
    / "data"
    / "podcasts"
    / "podcast_script.txt"
)

AUDIO_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "podcasts"
    / "audio"
)

TEMP_FOLDER = AUDIO_FOLDER / "temporary_clips"
FINAL_AUDIO_PATH = AUDIO_FOLDER / "podcast_episode.wav"

TTS_MODEL = "gpt-4o-mini-tts"

VOICE_SETTINGS = {
    "Alex": {
        "voice": "cedar",
        "instructions": (
            "Speak as Alex, a calm, warm and confident educational "
            "podcast presenter. Use clear British English pronunciation. "
            "Sound natural and engaging, not robotic. Explain ideas at "
            "a steady pace and use gentle emphasis on important terms."
        ),
    },
    "Jamie": {
        "voice": "marin",
        "instructions": (
            "Speak as Jamie, a friendly, curious and energetic podcast "
            "co-host. Use clear British English pronunciation. Sound "
            "interested and conversational. Ask questions naturally "
            "and react warmly to explanations without exaggerating."
        ),
    },
}


@dataclass
class DialogueTurn:
    speaker: str
    text: str


def clean_dialogue_text(text: str) -> str:
    """Remove unnecessary markdown while preserving natural speech."""

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\[(.*?)\]", r"\1", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_podcast_script(script: str) -> list[DialogueTurn]:
    """
    Extract dialogue spoken by Alex and Jamie.

    Supports formats such as:
    Alex: Hello
    **Alex:** Hello
    """

    speaker_pattern = re.compile(
        r"^\s*(?:\*\*)?(Alex|Jamie)(?:\*\*)?\s*:\s*(.*)$",
        flags=re.IGNORECASE,
    )

    turns: list[DialogueTurn] = []
    current_speaker: str | None = None
    current_lines: list[str] = []

    def save_current_turn() -> None:
        nonlocal current_speaker, current_lines

        if not current_speaker:
            return

        text = clean_dialogue_text(" ".join(current_lines))

        if text:
            normalised_speaker = current_speaker.capitalize()

            turns.append(
                DialogueTurn(
                    speaker=normalised_speaker,
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
            # Continue the current host's speech across multiple lines.
            current_lines.append(line)

    save_current_turn()

    if not turns:
        raise ValueError(
            "No Alex or Jamie dialogue was detected in the podcast script."
        )

    return turns


def split_long_text(
    text: str,
    character_limit: int = 3800,
) -> list[str]:
    """Split long dialogue safely below the TTS input limit."""

    if len(text) <= character_limit:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        if not sentence:
            continue

        proposed = (
            f"{current_chunk} {sentence}".strip()
            if current_chunk
            else sentence
        )

        if len(proposed) <= character_limit:
            current_chunk = proposed
            continue

        if current_chunk:
            chunks.append(current_chunk)

        # Handle an unusually long sentence.
        while len(sentence) > character_limit:
            chunks.append(sentence[:character_limit])
            sentence = sentence[character_limit:]

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
    """Generate one WAV speech clip."""

    settings = VOICE_SETTINGS[speaker]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    """Create silent WAV frames matching the generated audio."""

    frame_count = int(
        wav_parameters.framerate * duration_seconds
    )

    bytes_per_frame = (
        wav_parameters.sampwidth
        * wav_parameters.nchannels
    )

    return b"\x00" * frame_count * bytes_per_frame


def combine_wav_files(
    clip_paths: list[Path],
    output_path: Path,
    pause_seconds: float = 0.45,
) -> None:
    """Combine WAV clips and insert silence between speakers."""

    if not clip_paths:
        raise ValueError("No audio clips were generated.")

    with wave.open(str(clip_paths[0]), "rb") as first_clip:
        expected_parameters = first_clip.getparams()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with wave.open(str(output_path), "wb") as output_wav:
        output_wav.setnchannels(
            expected_parameters.nchannels
        )
        output_wav.setsampwidth(
            expected_parameters.sampwidth
        )
        output_wav.setframerate(
            expected_parameters.framerate
        )

        pause = silence_frames(
            expected_parameters,
            pause_seconds,
        )

        for index, clip_path in enumerate(clip_paths):
            with wave.open(str(clip_path), "rb") as clip:
                clip_parameters = clip.getparams()

                comparable_expected = (
                    expected_parameters.nchannels,
                    expected_parameters.sampwidth,
                    expected_parameters.framerate,
                    expected_parameters.comptype,
                )

                comparable_current = (
                    clip_parameters.nchannels,
                    clip_parameters.sampwidth,
                    clip_parameters.framerate,
                    clip_parameters.comptype,
                )

                if comparable_current != comparable_expected:
                    raise ValueError(
                        f"Incompatible WAV settings in {clip_path.name}."
                    )

                output_wav.writeframes(
                    clip.readframes(clip.getnframes())
                )

            if index < len(clip_paths) - 1:
                output_wav.writeframes(pause)


def main() -> None:
    load_dotenv(dotenv_path=ENV_PATH)

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY was not found.")
        return

    if not SCRIPT_PATH.exists():
        print("❌ Podcast script was not found.")
        print("Run backend/podcast_generator.py first.")
        return

    script = SCRIPT_PATH.read_text(encoding="utf-8")

    if not script.strip():
        print("❌ The podcast script is empty.")
        return

    try:
        dialogue_turns = parse_podcast_script(script)

        print(
            f"Detected {len(dialogue_turns)} dialogue turns."
        )

        client = OpenAI(api_key=api_key)

        # Remove clips left behind by an earlier attempt.
        if TEMP_FOLDER.exists():
            shutil.rmtree(TEMP_FOLDER)

        TEMP_FOLDER.mkdir(
            parents=True,
            exist_ok=True,
        )

        clip_paths: list[Path] = []
        clip_number = 1

        for turn_number, turn in enumerate(
            dialogue_turns,
            start=1,
        ):
            chunks = split_long_text(turn.text)

            for chunk_number, chunk in enumerate(
                chunks,
                start=1,
            ):
                print(
                    f"Recording turn {turn_number}/"
                    f"{len(dialogue_turns)} — "
                    f"{turn.speaker}"
                )

                clip_path = (
                    TEMP_FOLDER
                    / (
                        f"{clip_number:04d}_"
                        f"{turn.speaker.lower()}_"
                        f"{chunk_number}.wav"
                    )
                )

                generate_voice_clip(
                    client=client,
                    speaker=turn.speaker,
                    text=chunk,
                    output_path=clip_path,
                )

                clip_paths.append(clip_path)
                clip_number += 1

        print("\nCombining Alex and Jamie's audio...")

        combine_wav_files(
            clip_paths=clip_paths,
            output_path=FINAL_AUDIO_PATH,
            pause_seconds=0.45,
        )

        shutil.rmtree(
            TEMP_FOLDER,
            ignore_errors=True,
        )

        print("\n✅ Two-voice podcast created!")
        print(f"Saved to: {FINAL_AUDIO_PATH}")
        print(
            "\nDisclosure: This episode uses "
            "AI-generated voices."
        )

    except Exception as error:
        print(f"\n❌ Podcast audio generation failed: {error}")


if __name__ == "__main__":
    main()