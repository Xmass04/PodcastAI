from __future__ import annotations

import json
import os
import re
import tempfile
import time
import wave
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
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

MAX_PARALLEL_REQUESTS = max(
    1,
    min(
        int(os.getenv("PODCASTAI_TTS_WORKERS", "2")),
        6,
    ),
)

MAX_TTS_ATTEMPTS = 3
TTS_REQUEST_TIMEOUT_SECONDS = max(
    30.0,
    float(
        os.getenv(
            "PODCASTAI_TTS_TIMEOUT_SECONDS",
            "120",
        )
    ),
)
TEXT_CHARACTER_LIMIT = 3400

VOICE_SETTINGS = {
    "Alex": {
        "voice": "cedar",
        "instructions": (
            "Speak as Alex, a calm, warm and confident presenter. "
            "Use clear British English pronunciation. Sound natural, "
            "engaging and organised."
        ),
    },
    "Jamie": {
        "voice": "marin",
        "instructions": (
            "Speak as Jamie, a friendly, curious and conversational "
            "co-host. Use clear British English pronunciation. "
            "Sound interested and natural."
        ),
    },
    "Narrator": {
        "voice": "cedar",
        "instructions": (
            "Speak as a warm, expressive story narrator. Use clear "
            "British English pronunciation, natural dramatic pacing "
            "and gentle emotional variation."
        ),
    },
}


@dataclass(frozen=True)
class DialogueTurn:
    speaker: str
    text: str


@dataclass(frozen=True)
class AudioClipPlan:
    index: int
    speaker: str
    text: str
    output_path: Path


def load_analysis() -> dict[str, Any]:
    if not ANALYSIS_PATH.exists():
        return {"document_mode": "study"}

    try:
        return json.loads(
            ANALYSIS_PATH.read_text(
                encoding="utf-8",
                errors="replace",
            )
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


def parse_audio_script(
    script: str,
    document_mode: str,
) -> list[DialogueTurn]:
    speaker_pattern = re.compile(
        r"^\s*(?:\*\*)?"
        r"(Alex|Jamie|Narrator)"
        r"(?:\*\*)?\s*:\s*(.*)$",
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

        text = clean_dialogue_text(
            " ".join(current_lines)
        )

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
            if not re.match(
                r"^(title|chapter|section)\b",
                line,
                flags=re.IGNORECASE,
            ):
                unlabelled_story_lines.append(line)

    save_current_turn()

    if (
        document_mode == "story"
        and unlabelled_story_lines
        and not turns
    ):
        story_text = clean_dialogue_text(
            " ".join(unlabelled_story_lines)
        )

        if story_text:
            turns.append(
                DialogueTurn(
                    speaker="Narrator",
                    text=story_text,
                )
            )

    if not turns:
        raise ValueError(
            "No supported spoken content was detected. "
            "Study, work and research scripts need Alex: and Jamie: "
            "labels. Story scripts need Narrator: labels or prose."
        )

    return turns


def merge_consecutive_turns(
    turns: list[DialogueTurn],
) -> list[DialogueTurn]:
    """
    Merge only genuinely consecutive turns from the same speaker.

    Speaker order is preserved exactly. Alternating Alex/Jamie turns are
    never merged because that would change the conversation.
    """

    if not turns:
        return []

    merged: list[DialogueTurn] = []

    for turn in turns:
        if (
            merged
            and merged[-1].speaker == turn.speaker
            and len(merged[-1].text) + len(turn.text) + 1
            <= TEXT_CHARACTER_LIMIT
        ):
            previous = merged[-1]

            merged[-1] = DialogueTurn(
                speaker=previous.speaker,
                text=f"{previous.text} {turn.text}".strip(),
            )
        else:
            merged.append(turn)

    return merged


def split_long_text(
    text: str,
    character_limit: int = TEXT_CHARACTER_LIMIT,
) -> list[str]:
    if len(text) <= character_limit:
        return [text]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()

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

        while len(sentence) > character_limit:
            split_point = sentence.rfind(
                " ",
                0,
                character_limit,
            )

            if split_point <= 0:
                split_point = character_limit

            chunks.append(
                sentence[:split_point].strip()
            )

            sentence = sentence[
                split_point:
            ].strip()

        current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def build_clip_plan(
    turns: list[DialogueTurn],
    temp_folder: Path,
) -> list[AudioClipPlan]:
    plans: list[AudioClipPlan] = []
    clip_index = 1

    for turn in turns:
        for chunk_number, chunk in enumerate(
            split_long_text(turn.text),
            start=1,
        ):
            output_path = (
                temp_folder
                / (
                    f"{clip_index:04d}_"
                    f"{turn.speaker.lower()}_"
                    f"{chunk_number}.wav"
                )
            )

            plans.append(
                AudioClipPlan(
                    index=clip_index,
                    speaker=turn.speaker,
                    text=chunk,
                    output_path=output_path,
                )
            )

            clip_index += 1

    return plans


def generate_voice_clip(
    api_key: str,
    plan: AudioClipPlan,
) -> Path:
    settings = VOICE_SETTINGS[plan.speaker]

    plan.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_TTS_ATTEMPTS + 1,
    ):
        try:
            client = OpenAI(
                api_key=api_key,
                timeout=TTS_REQUEST_TIMEOUT_SECONDS,
                max_retries=0,
            )

            response = client.audio.speech.create(
                model=TTS_MODEL,
                voice=settings["voice"],
                input=plan.text,
                instructions=settings["instructions"],
                response_format="wav",
                speed=1.0,
                timeout=TTS_REQUEST_TIMEOUT_SECONDS,
            )

            plan.output_path.write_bytes(
                response.content
            )

            if (
                not plan.output_path.exists()
                or plan.output_path.stat().st_size == 0
            ):
                raise RuntimeError(
                    "The generated voice clip was empty."
                )

            client.close()
            return plan.output_path

        except Exception as error:
            last_error = error

            try:
                client.close()
            except Exception:
                pass

            if plan.output_path.exists():
                try:
                    plan.output_path.unlink()
                except OSError:
                    pass

            print(
                f"PROGRESS|5|Clip {plan.index} timed out or failed "
                f"(attempt {attempt}/{MAX_TTS_ATTEMPTS}); retrying...",
                flush=True,
            )

            if attempt >= MAX_TTS_ATTEMPTS:
                break

            time.sleep(min(2 ** attempt, 5))

    raise RuntimeError(
        f"Clip {plan.index} for {plan.speaker} failed "
        f"after {MAX_TTS_ATTEMPTS} attempts: {last_error}"
    )


def generate_clips_in_parallel(
    api_key: str,
    plans: list[AudioClipPlan],
) -> list[Path]:
    if not plans:
        raise ValueError(
            "No audio clips were planned."
        )

    completed_paths: dict[int, Path] = {}
    completed_count = 0
    total = len(plans)

    print(
        f"PROGRESS|5|Generating {total} audio clips "
        f"with {MAX_PARALLEL_REQUESTS} stable workers "
        f"and a {int(TTS_REQUEST_TIMEOUT_SECONDS)}s clip timeout...",
        flush=True,
    )

    with ThreadPoolExecutor(
        max_workers=MAX_PARALLEL_REQUESTS
    ) as executor:
        future_map: dict[
            Future[Path],
            AudioClipPlan,
        ] = {
            executor.submit(
                generate_voice_clip,
                api_key,
                plan,
            ): plan
            for plan in plans
        }

        for future in as_completed(future_map):
            plan = future_map[future]

            try:
                output_path = future.result()
            except Exception:
                for pending_future in future_map:
                    pending_future.cancel()

                raise

            completed_paths[
                plan.index
            ] = output_path

            completed_count += 1

            percent = min(
                90,
                5
                + int(
                    completed_count
                    / total
                    * 85
                ),
            )

            print(
                f"Recording {completed_count}/{total} "
                f"- {plan.speaker}",
                flush=True,
            )

            print(
                f"PROGRESS|{percent}|"
                f"Recorded clip {completed_count}/{total}",
                flush=True,
            )

    return [
        completed_paths[index]
        for index in sorted(completed_paths)
    ]


def silence_frames(
    wav_parameters: wave._wave_params,
    duration_seconds: float,
) -> bytes:
    frame_count = int(
        wav_parameters.framerate
        * duration_seconds
    )

    bytes_per_frame = (
        wav_parameters.sampwidth
        * wav_parameters.nchannels
    )

    return (
        b"\x00"
        * frame_count
        * bytes_per_frame
    )


def combine_wav_files(
    clip_paths: list[Path],
    output_path: Path,
    pause_seconds: float,
) -> None:
    if not clip_paths:
        raise ValueError(
            "No voice clips were generated."
        )

    with wave.open(
        str(clip_paths[0]),
        "rb",
    ) as first_clip:
        expected = first_clip.getparams()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    building_path = output_path.with_name(
        f"{output_path.stem}_building.wav"
    )

    if building_path.exists():
        building_path.unlink()

    with wave.open(
        str(building_path),
        "wb",
    ) as output_wav:
        output_wav.setnchannels(
            expected.nchannels
        )
        output_wav.setsampwidth(
            expected.sampwidth
        )
        output_wav.setframerate(
            expected.framerate
        )

        pause = silence_frames(
            expected,
            pause_seconds,
        )

        for index, clip_path in enumerate(
            clip_paths
        ):
            with wave.open(
                str(clip_path),
                "rb",
            ) as clip:
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
                    raise ValueError(
                        "Incompatible WAV settings in "
                        f"{clip_path.name}."
                    )

                output_wav.writeframes(
                    clip.readframes(
                        clip.getnframes()
                    )
                )

            if index < len(clip_paths) - 1:
                output_wav.writeframes(pause)

    try:
        os.replace(
            building_path,
            output_path,
        )
    except PermissionError as error:
        raise PermissionError(
            "The existing podcast file is open. "
            "Close the audio player or browser tab "
            "and generate it again."
        ) from error


def main() -> None:
    load_dotenv(dotenv_path=ENV_PATH)

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY was not found."
        )
        raise SystemExit(1)

    if not SCRIPT_PATH.exists():
        print(
            "ERROR: Podcast script was not found."
        )
        print(
            "Run backend/podcast_generator.py first."
        )
        raise SystemExit(1)

    script = SCRIPT_PATH.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    if not script:
        print(
            "ERROR: The podcast script is empty."
        )
        raise SystemExit(1)

    try:
        print(
            "PROGRESS|1|Parsing podcast script...",
            flush=True,
        )

        print(
            "Using stable non-streaming TTS mode.",
            flush=True,
        )

        analysis = load_analysis()

        document_mode = str(
            analysis.get(
                "document_mode",
                "study",
            )
        ).lower()

        parsed_turns = parse_audio_script(
            script,
            document_mode,
        )

        merged_turns = merge_consecutive_turns(
            parsed_turns
        )

        print(
            f"Detected {len(parsed_turns)} spoken turns "
            f"for {document_mode} mode.",
            flush=True,
        )

        print(
            f"Prepared {len(merged_turns)} ordered "
            "speaker sections.",
            flush=True,
        )

        with tempfile.TemporaryDirectory(
            prefix="podcastai_audio_",
            ignore_cleanup_errors=True,
        ) as temporary_directory:
            temp_folder = Path(
                temporary_directory
            )

            plans = build_clip_plan(
                merged_turns,
                temp_folder,
            )

            print(
                f"Prepared {len(plans)} TTS clips.",
                flush=True,
            )

            clip_paths = (
                generate_clips_in_parallel(
                    api_key,
                    plans,
                )
            )

            pause_seconds = (
                0.70
                if document_mode == "story"
                else 0.55
            )

            print(
                "PROGRESS|95|Combining audio...",
                flush=True,
            )

            print(
                "\nCombining generated audio...",
                flush=True,
            )

            combine_wav_files(
                clip_paths,
                FINAL_AUDIO_PATH,
                pause_seconds,
            )

        print(
            "PROGRESS|100|Podcast audio ready",
            flush=True,
        )

        print(
            "\nSUCCESS: Audio created!"
        )

        print(
            f"Saved to: {FINAL_AUDIO_PATH}"
        )

        print(
            "Disclosure: This audio uses "
            "AI-generated voices."
        )

    except Exception as error:
        print(
            "\nERROR: Podcast audio generation "
            f"failed: {error}"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()