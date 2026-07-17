from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st


BACKEND_IMPORT_FOLDER = Path(__file__).resolve().parent.parent / "backend"

if str(BACKEND_IMPORT_FOLDER) not in sys.path:
    sys.path.insert(0, str(BACKEND_IMPORT_FOLDER))

from diagnostics import (
    load_latest_report,
    record_python_exception,
    record_subprocess_result,
    utc_now,
    wait_before_retry,
)


from cache_manager import (
    build_cache_key,
    cache_exists,
    calculate_file_hash,
    restore_cache,
    save_cache,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_FOLDER = PROJECT_ROOT / "backend"
DATA_FOLDER = PROJECT_ROOT / "data"

UPLOAD_FOLDER = DATA_FOLDER / "uploads"
IMAGE_FOLDER = DATA_FOLDER / "images"

EXTRACTED_TEXT_PATH = DATA_FOLDER / "extracted_text" / "output.txt"
ANALYSIS_PATH = DATA_FOLDER / "analysis" / "analysis.json"
SUMMARY_PATH = DATA_FOLDER / "summaries" / "summary.txt"
NOTES_PATH = DATA_FOLDER / "summaries" / "study_notes.txt"
FLASHCARDS_PATH = DATA_FOLDER / "flashcards" / "flashcards.json"
QUIZ_PATH = DATA_FOLDER / "quizzes" / "quiz.json"
PODCAST_SCRIPT_PATH = DATA_FOLDER / "podcasts" / "podcast_script.txt"
PODCAST_AUDIO_PATH = DATA_FOLDER / "podcasts" / "audio" / "podcast_episode.wav"
PODCAST_SOURCE_HASH_PATH = DATA_FOLDER / "podcasts" / "source_hash.txt"

TASKS = {
    "Document analysis": "document_analyzer.py",
    "Summary": "ai_summarizer.py",
    "Study notes": "study_notes.py",
    "Flashcards": "flashcard_generator.py",
    "Quiz": "quiz_generator.py",
    "Podcast script": "podcast_generator.py",
    "Podcast audio": "podcast_audio.py",
}

TASK_ORDER = list(TASKS.keys())

MODE_LABELS = {
    "study": "Study Material",
    "story": "Story or Novel",
    "work": "Work Document",
    "research": "Research Paper",
    "general": "General Document",
}

MODE_PACKS = {
    "study": "Study Pack",
    "story": "Story Pack",
    "work": "Work Brief",
    "research": "Research Pack",
    "general": "General Summary",
}

MODE_AUDIO_STYLES = {
    "study": "Tutor Conversation",
    "story": "Story Narration",
    "work": "Professional Briefing",
    "research": "Research Discussion",
    "general": "General Explanation",
}


def configure_page() -> None:
    st.set_page_config(
        page_title="PodcastAI",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        .hero {
            text-align: center;
            padding: 2rem 1rem 1.25rem;
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 760;
            margin-bottom: 0.35rem;
        }
        .hero-text {
            font-size: 1.15rem;
            color: #9aa4b2;
        }
        .info-card {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin: 0.75rem 0;
        }
        div.stButton > button {
            width: 100%;
            min-height: 3rem;
            border-radius: 12px;
            font-weight: 650;
        }
        div[data-testid="stExpander"] {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialise_state() -> None:
    defaults = {
        "material_ready": False,
        "uploaded_name": None,
        "generation_complete": False,
        "detected_mode": "study",
        "selected_mode": "study",
        "developer_mode": False,
        "current_file_hash": None,
        "current_cache_key": None,
        "cache_loaded": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def run_backend_script(
    script_name: str,
    input_text: str | None = None,
    extra_environment: dict[str, str] | None = None,
    stage_name: str | None = None,
    max_attempts: int = 2,
) -> tuple[bool, str]:
    """
    Run one backend script with diagnostics and safe recovery.

    Retryable failures such as temporary timeouts and rate limits are
    retried once. Source code is never changed automatically.
    """

    script_path = BACKEND_FOLDER / script_name
    stage = stage_name or script_path.stem.replace("_", " ").title()

    if not script_path.exists():
        message = f"Missing backend file: {script_path}"

        started_at = utc_now()

        report = record_python_exception(
            stage=stage,
            script_name=script_name,
            started_at=started_at,
            error=FileNotFoundError(message),
        )

        return (
            False,
            (
                f"{message}\n\n"
                f"Suggested fix: {report.suggested_fix or 'Check the file path.'}"
            ),
        )

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"

    if extra_environment:
        environment.update(extra_environment)

    final_output = ""

    for attempt in range(1, max_attempts + 1):
        started_at = utc_now()

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=PROJECT_ROOT,
                input=input_text,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                check=False,
            )

        except OSError as error:
            report = record_python_exception(
                stage=stage,
                script_name=script_name,
                started_at=started_at,
                error=error,
                attempt=attempt,
            )

            final_output = (
                f"{error}\n\n"
                f"Likely cause: {report.likely_cause}\n"
                f"Suggested fix: {report.suggested_fix}"
            )

            if report.retryable and attempt < max_attempts:
                wait_before_retry(attempt)
                continue

            return False, final_output

        finished_at = utc_now()

        report = record_subprocess_result(
            stage=stage,
            script_name=script_name,
            started_at=started_at,
            finished_at=finished_at,
            return_code=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            attempt=attempt,
            metadata={
                "input_supplied": input_text is not None,
                "environment_overrides": sorted(
                    (extra_environment or {}).keys()
                ),
            },
        )

        output = "\n".join(
            part
            for part in (result.stdout, result.stderr)
            if part
        ).strip()

        if report.status == "success":
            return True, output

        location_parts = []

        if report.source_file:
            location_parts.append(
                f"File: {report.source_file}"
            )

        if report.source_function:
            location_parts.append(
                f"Function: {report.source_function}"
            )

        if report.source_line:
            location_parts.append(
                f"Line: {report.source_line}"
            )

        diagnostic_summary = [
            output,
            "",
            f"Diagnostic type: {report.error_type}",
            f"Likely cause: {report.likely_cause}",
            f"Suggested fix: {report.suggested_fix}",
        ]

        if location_parts:
            diagnostic_summary.append(
                "Location: " + " | ".join(location_parts)
            )

        final_output = "\n".join(
            item
            for item in diagnostic_summary
            if item is not None
        ).strip()

        if report.retryable and attempt < max_attempts:
            wait_before_retry(attempt)
            continue

        return False, final_output

    return False, final_output or "The backend task failed."


def clear_folder(
    folder: Path,
    attempts: int = 4,
) -> list[Path]:
    """
    Clear generated files without crashing the app when Windows or OneDrive
    temporarily locks an item.

    Returns any items that could not be removed. The caller may decide whether
    those locked items are safe to ignore.
    """

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    locked_items: list[Path] = []

    for item in list(folder.iterdir()):
        removed = False

        for attempt in range(1, attempts + 1):
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink(
                        missing_ok=True
                    )
                elif item.is_dir():
                    shutil.rmtree(item)

                removed = True
                break

            except (
                PermissionError,
                OSError,
            ):
                if attempt < attempts:
                    time.sleep(
                        0.35 * attempt
                    )

        if not removed and item.exists():
            locked_items.append(item)

    return locked_items


def clear_previous_project_data() -> None:
    """
    Clear document-specific outputs before analysing a new upload.

    The podcasts folder is intentionally excluded because Windows, OneDrive,
    the browser or an audio player may still have the previous WAV file open.
    Stale podcast audio is hidden using the current-upload source hash and is
    replaced only when the user explicitly generates a new podcast.
    """

    folders = [
        IMAGE_FOLDER,
        DATA_FOLDER / "extracted_text",
        DATA_FOLDER / "analysis",
        DATA_FOLDER / "summaries",
        DATA_FOLDER / "flashcards",
        DATA_FOLDER / "quizzes",
    ]

    locked_items: list[Path] = []

    for folder in folders:
        locked_items.extend(
            clear_folder(folder)
        )

    if locked_items:
        st.warning(
            "Some old generated files are temporarily locked by Windows or "
            "OneDrive. PodcastAI will continue with the new upload, but those "
            "locked files may remain on disk until the other program releases "
            "them."
        )

    st.session_state.material_ready = False
    st.session_state.generation_complete = False
    st.session_state.detected_mode = "study"
    st.session_state.selected_mode = "study"
    st.session_state.current_cache_key = None
    st.session_state.cache_loaded = False


def save_uploaded_file(uploaded_file: Any) -> Path:
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    safe_name = Path(uploaded_file.name).name
    destination = UPLOAD_FOLDER / safe_name
    destination.write_bytes(uploaded_file.getbuffer())

    st.session_state.current_file_hash = calculate_file_hash(
        destination
    )

    return destination


def extract_uploaded_material(uploaded_file: Any) -> tuple[bool, str]:
    clear_previous_project_data()

    uploaded_path = save_uploaded_file(uploaded_file)
    extension = uploaded_path.suffix.lower()

    if extension == ".pdf":
        return run_backend_script(
            "pdf_reader.py",
            input_text=f"{uploaded_path}\n",
        )

    if extension in {".jpg", ".jpeg", ".png", ".webp"}:
        IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
        destination = IMAGE_FOLDER / uploaded_path.name
        shutil.copy2(uploaded_path, destination)
        return run_backend_script("image_reader.py")

    return False, "Unsupported file type."


def read_text_file(path: Path) -> str | None:
    if not path.exists():
        return None

    content = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    return content or None


def read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )
    except (json.JSONDecodeError, OSError):
        return None


def save_analysis(analysis: dict[str, Any]) -> None:
    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def apply_mode_override(selected_mode: str) -> None:
    analysis = read_json_file(ANALYSIS_PATH) or {}

    analysis["document_mode"] = selected_mode
    analysis["recommended_pack"] = MODE_PACKS[selected_mode]
    analysis["audio_style"] = MODE_AUDIO_STYLES[selected_mode]
    analysis["user_overrode_mode"] = (
        selected_mode != st.session_state.detected_mode
    )

    save_analysis(analysis)


def analyse_current_material() -> tuple[bool, str]:
    success, output = run_backend_script("document_analyzer.py")

    if not success:
        return success, output

    analysis = read_json_file(ANALYSIS_PATH)
    if not analysis:
        return False, "analysis.json was not created."

    detected_mode = str(
        analysis.get("document_mode", "study")
    ).lower()

    if detected_mode not in MODE_LABELS:
        detected_mode = "general"

    st.session_state.detected_mode = detected_mode
    st.session_state.selected_mode = detected_mode

    return True, output


def current_mode() -> str:
    mode = st.session_state.get("selected_mode", "study")
    return mode if mode in MODE_LABELS else "general"


def prepare_tasks(selected_tasks: list[str]) -> list[str]:
    requested = set(selected_tasks)

    if "Podcast audio" in requested:
        requested.discard("Podcast script")

    return [task for task in TASK_ORDER if task in requested]


def current_upload_hash() -> str | None:
    value = st.session_state.get(
        "current_file_hash"
    )

    return str(value) if value else None


def clear_podcast_artifacts() -> None:
    """
    Remove the previous script, audio and source marker before creating a new
    podcast. This prevents a failed generation from exposing an older episode.
    """

    paths = [
        PODCAST_SCRIPT_PATH,
        PODCAST_AUDIO_PATH,
        PODCAST_SOURCE_HASH_PATH,
    ]

    for path in paths:
        if not path.exists():
            continue

        for attempt in range(1, 5):
            try:
                path.unlink(
                    missing_ok=True
                )
                break
            except (
                PermissionError,
                OSError,
            ) as error:
                if attempt >= 4:
                    raise PermissionError(
                        f"Could not replace the old podcast file: {path}. "
                        "Close any open audio player or preview and retry."
                    ) from error

                time.sleep(
                    0.4 * attempt
                )


def mark_podcast_for_current_upload() -> None:
    file_hash = current_upload_hash()

    if not file_hash:
        return

    PODCAST_SOURCE_HASH_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PODCAST_SOURCE_HASH_PATH.write_text(
        file_hash,
        encoding="utf-8",
    )


def podcast_matches_current_upload() -> bool:
    file_hash = current_upload_hash()

    if (
        not file_hash
        or not PODCAST_SOURCE_HASH_PATH.exists()
    ):
        return False

    saved_hash = PODCAST_SOURCE_HASH_PATH.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    return saved_hash == file_hash


def generate_podcast_audio_task(
    mode: str,
    podcast_length: str,
) -> tuple[bool, str]:
    clear_podcast_artifacts()
    outputs: list[str] = []

    if mode == "study":
        success, output = run_backend_script("study_notes.py")
        outputs.append(output)

        if not success or not read_text_file(NOTES_PATH):
            return False, f"Study-note generation failed.\n\n{output}"

    elif mode in {"work", "research", "general"}:
        success, output = run_backend_script("ai_summarizer.py")
        outputs.append(output)

        if not success or not read_text_file(SUMMARY_PATH):
            return False, f"Summary generation failed.\n\n{output}"

    script_success, script_output = run_backend_script(
        "podcast_generator.py",
        extra_environment={
            "PODCASTAI_PODCAST_LENGTH": podcast_length,
        },
        stage_name="Podcast Script",
    )
    outputs.append(script_output)

    if not script_success or not read_text_file(PODCAST_SCRIPT_PATH):
        return False, f"Podcast script generation failed.\n\n{script_output}"

    audio_success, audio_output = run_backend_script("podcast_audio.py")
    outputs.append(audio_output)

    if not audio_success or not PODCAST_AUDIO_PATH.exists():
        return False, f"Podcast audio generation failed.\n\n{audio_output}"

    mark_podcast_for_current_upload()

    return True, "\n\n".join(output for output in outputs if output)


def show_live_result(task_name: str, container: Any) -> None:
    with container:
        if task_name == "Document analysis":
            st.success("Document analysis ready.")

        elif task_name == "Summary":
            content = read_text_file(SUMMARY_PATH)
            st.success("Summary ready — you can start reading now.")

            if content:
                with st.expander("Read summary now", expanded=True):
                    st.markdown(content)

        elif task_name == "Study notes":
            content = read_text_file(NOTES_PATH)
            st.success("Study notes ready.")

            if content:
                with st.expander("Read study notes now"):
                    st.markdown(content)

        elif task_name == "Flashcards":
            data = read_json_file(FLASHCARDS_PATH) or {}
            cards = data.get("flashcards", [])
            st.success(f"Flashcards ready ({len(cards)} cards).")

        elif task_name == "Quiz":
            data = read_json_file(QUIZ_PATH) or {}
            total = (
                len(data.get("multiple_choice", []))
                + len(data.get("true_false", []))
                + len(data.get("short_answer", []))
            )
            st.success(f"Quiz ready ({total} questions).")

        elif task_name == "Podcast script":
            script = read_text_file(PODCAST_SCRIPT_PATH)
            st.success("Audio script ready.")

            if script:
                with st.expander("Preview the script"):
                    st.text(script[:5000])

        elif task_name == "Podcast audio":
            if PODCAST_AUDIO_PATH.exists():
                st.success("Audio ready — press play.")
                st.audio(
                    PODCAST_AUDIO_PATH.read_bytes(),
                    format="audio/wav",
                )


def generate_tasks(
    selected_tasks: list[str],
    podcast_length: str,
) -> dict[str, tuple[bool, str]]:
    mode = current_mode()
    ordered_tasks = prepare_tasks(selected_tasks)
    results: dict[str, tuple[bool, str]] = {}

    st.markdown("## Live results")
    st.caption(
        "Completed resources appear while PodcastAI continues working."
    )

    containers = {task: st.container() for task in TASK_ORDER}
    progress = st.progress(0, text="Preparing...")

    with st.status("Building your content...", expanded=True) as status:
        total = len(ordered_tasks)

        for index, task_name in enumerate(ordered_tasks, start=1):
            st.write(f"Creating **{task_name}**...")

            if task_name == "Podcast audio":
                st.write("Preparing the required source, script and audio...")
                success, output = generate_podcast_audio_task(
                    mode,
                    podcast_length,
                )

                if success:
                    show_live_result(
                        "Podcast script",
                        containers["Podcast script"],
                    )
            else:
                if task_name == "Podcast script":
                    clear_podcast_artifacts()

                    success, output = run_backend_script(
                        TASKS[task_name],
                        extra_environment={
                            "PODCASTAI_PODCAST_LENGTH": podcast_length,
                        },
                        stage_name="Podcast Script",
                    )

                    if (
                        success
                        and read_text_file(
                            PODCAST_SCRIPT_PATH
                        )
                    ):
                        mark_podcast_for_current_upload()
                else:
                    success, output = run_backend_script(
                        TASKS[task_name],
                        stage_name=task_name,
                    )

            results[task_name] = (success, output)

            if success:
                st.write(f"✅ {task_name} ready")
                show_live_result(task_name, containers[task_name])
            else:
                st.write(f"⚠️ {task_name} failed")

                with containers[task_name]:
                    st.error(f"{task_name} could not be completed.")

                    if output:
                        with st.expander("Technical details"):
                            st.code(output)

            progress.progress(
                int(index / total * 100),
                text=f"{task_name}: {'ready' if success else 'failed'}",
            )

        successful = sum(
            1 for success, _ in results.values() if success
        )

        if successful == total:
            status.update(
                label="Your requested content is ready!",
                state="complete",
                expanded=False,
            )
        else:
            status.update(
                label=f"Finished with {successful}/{total} items created",
                state="complete",
                expanded=True,
            )

    return results


def display_analysis_card() -> None:
    analysis = read_json_file(ANALYSIS_PATH)
    if not analysis:
        return

    detected_mode = st.session_state.detected_mode
    confidence = analysis.get("confidence", "Unknown")
    document_type = analysis.get("document_type", "Unknown")
    title = analysis.get("document_title", "Uploaded document")

    st.markdown("## Document detected")

    first, second, third = st.columns(3)
    first.metric("Mode", MODE_LABELS[detected_mode])
    second.metric("Document type", document_type)
    third.metric("Confidence", f"{confidence}%")

    st.caption(title)

    selected_mode = st.selectbox(
        "Use this document as",
        options=list(MODE_LABELS.keys()),
        format_func=lambda value: MODE_LABELS[value],
        index=list(MODE_LABELS.keys()).index(
            st.session_state.selected_mode
        ),
        help="Change this when PodcastAI detects the wrong type.",
    )

    if selected_mode != st.session_state.selected_mode:
        st.session_state.selected_mode = selected_mode
        apply_mode_override(selected_mode)
        st.success(f"Document mode updated to {MODE_LABELS[selected_mode]}.")

    mode = current_mode()
    st.info(
        f"Recommended: {MODE_PACKS[mode]} • "
        f"Audio style: {MODE_AUDIO_STYLES[mode]}"
    )


def display_text_result(title: str, path: Path) -> None:
    content = read_text_file(path)
    st.subheader(title)

    if not content:
        st.info(f"{title} has not been generated yet.")
        return

    st.markdown(content)


def display_flashcards() -> None:
    st.subheader("Flashcards")
    data = read_json_file(FLASHCARDS_PATH)

    if not data:
        st.info("Flashcards have not been generated yet.")
        return

    cards = data.get("flashcards", [])
    st.caption(f"{len(cards)} cards — open one to reveal the answer.")

    for index, card in enumerate(cards, start=1):
        question = card.get("question", "Question unavailable")

        with st.expander(f"Card {index}: {question}"):
            st.write(card.get("answer", "Answer unavailable"))

            tip = card.get("exam_tip", "")
            if tip:
                st.info(f"Tip: {tip}")


def display_quiz() -> None:
    st.subheader("Quiz")
    data = read_json_file(QUIZ_PATH)

    if not data:
        st.info("The quiz has not been generated yet.")
        return

    multiple_choice = data.get("multiple_choice", [])
    true_false = data.get("true_false", [])
    short_answer = data.get("short_answer", [])

    with st.form("podcastai_quiz"):
        mc_answers = []
        tf_answers = []
        written_answers = []

        for index, question in enumerate(multiple_choice, start=1):
            mc_answers.append(
                st.radio(
                    f"{index}. {question.get('question', '')}",
                    question.get("options", []),
                    index=None,
                    key=f"mc_{index}",
                )
            )

        for index, question in enumerate(true_false, start=1):
            tf_answers.append(
                st.radio(
                    f"{index}. {question.get('statement', '')}",
                    ["True", "False"],
                    index=None,
                    key=f"tf_{index}",
                )
            )

        for index, question in enumerate(short_answer, start=1):
            written_answers.append(
                st.text_area(
                    f"{index}. {question.get('question', '')}",
                    key=f"short_{index}",
                )
            )

        submitted = st.form_submit_button(
            "Submit Quiz",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    score = 0
    marked_total = len(multiple_choice) + len(true_false)
    st.header("Your Results")

    for index, question in enumerate(multiple_choice, start=1):
        correct = question.get("correct_answer", "")

        if mc_answers[index - 1] == correct:
            score += 1
            st.success(f"Question {index}: Correct")
        else:
            st.error(f"Question {index}: Correct answer: {correct}")

    offset = len(multiple_choice)

    for index, question in enumerate(true_false, start=1):
        expected = "True" if question.get("correct_answer") else "False"
        number = offset + index

        if tf_answers[index - 1] == expected:
            score += 1
            st.success(f"Question {number}: Correct")
        else:
            st.error(f"Question {number}: Correct answer: {expected}")

    if marked_total:
        percentage = round(score / marked_total * 100)
        st.metric("Score", f"{score}/{marked_total}", f"{percentage}%")

    for index, question in enumerate(short_answer, start=1):
        st.markdown(f"**{index}. {question.get('question', '')}**")
        st.write(f"Your answer: {written_answers[index - 1] or 'No answer'}")
        st.info(f"Model answer: {question.get('model_answer', '')}")


def display_audio() -> None:
    mode = current_mode()
    st.subheader("Story Audio" if mode == "story" else "Podcast")

    podcast_is_current = (
        podcast_matches_current_upload()
    )

    if (
        PODCAST_AUDIO_PATH.exists()
        and podcast_is_current
    ):
        audio = PODCAST_AUDIO_PATH.read_bytes()
        st.audio(audio, format="audio/wav")
        st.caption("This audio uses AI-generated voices.")

        st.download_button(
            "Download Audio",
            data=audio,
            file_name=(
                "story_audio.wav" if mode == "story" else "podcast_episode.wav"
            ),
            mime="audio/wav",
            use_container_width=True,
        )
    else:
        st.info(
            "Audio has not been generated for the current upload yet."
        )

    script = (
        read_text_file(PODCAST_SCRIPT_PATH)
        if podcast_is_current
        else None
    )

    if script:
        with st.expander("Read the audio script"):
            st.text(script)


def display_results() -> None:
    st.divider()
    st.header("Your complete results")

    tabs = st.tabs(
        ["Overview", "Study Notes", "Flashcards", "Quiz", "Audio"]
    )

    with tabs[0]:
        display_text_result("Summary", SUMMARY_PATH)

    with tabs[1]:
        display_text_result("Study Notes", NOTES_PATH)

    with tabs[2]:
        display_flashcards()

    with tabs[3]:
        display_quiz()

    with tabs[4]:
        display_audio()


def tasks_for_mode(
    mode: str,
    pack_type: str,
    include_audio: bool,
) -> list[str]:
    if mode == "study":
        if pack_type == "Quick Pack":
            tasks = [
                "Document analysis",
                "Summary",
                "Flashcards",
                "Quiz",
            ]
        else:
            tasks = [
                "Document analysis",
                "Summary",
                "Study notes",
                "Flashcards",
                "Quiz",
                "Podcast script",
            ]

    elif mode == "story":
        tasks = [
            "Document analysis",
            "Summary",
            "Podcast script",
        ]

    elif mode == "research":
        tasks = [
            "Document analysis",
            "Summary",
            "Study notes",
            "Podcast script",
        ]

    else:
        tasks = [
            "Document analysis",
            "Summary",
            "Podcast script",
        ]

    if include_audio:
        tasks.append("Podcast audio")

    return tasks




def current_cache_key(
    podcast_length: str,
) -> str | None:
    file_hash = st.session_state.get(
        "current_file_hash"
    )

    if not file_hash:
        return None

    return build_cache_key(
        file_hash=file_hash,
        document_mode=current_mode(),
        podcast_length=podcast_length,
    )


def try_restore_cached_pack(
    podcast_length: str,
) -> bool:
    cache_key = current_cache_key(
        podcast_length
    )

    if not cache_key or not cache_exists(
        cache_key
    ):
        return False

    metadata = restore_cache(
        cache_key
    )

    st.session_state.current_cache_key = cache_key
    st.session_state.cache_loaded = True
    st.session_state.generation_complete = True

    st.success(
        "Previously generated results were found "
        "and loaded from the smart cache."
    )

    created_at = metadata.get(
        "created_at"
    )

    if created_at:
        st.caption(
            f"Cached on: {created_at}"
        )

    return True


def save_current_pack_to_cache(
    *,
    podcast_length: str,
    generated_tasks: list[str],
) -> None:
    cache_key = current_cache_key(
        podcast_length
    )

    file_hash = st.session_state.get(
        "current_file_hash"
    )

    uploaded_name = st.session_state.get(
        "uploaded_name"
    )

    if (
        not cache_key
        or not file_hash
        or not uploaded_name
    ):
        return

    save_cache(
        cache_key=cache_key,
        original_filename=uploaded_name,
        file_hash=file_hash,
        document_mode=current_mode(),
        podcast_length=podcast_length,
        generated_tasks=generated_tasks,
    )

    st.session_state.current_cache_key = cache_key



def display_developer_panel() -> None:
    """
    Show the latest diagnostic report only when Developer Mode is enabled.
    """

    st.session_state.developer_mode = st.sidebar.checkbox(
        "Developer Mode",
        value=st.session_state.developer_mode,
        help=(
            "Shows technical diagnostics. Keep this disabled "
            "for normal beta users."
        ),
    )

    if not st.session_state.developer_mode:
        return

    latest = load_latest_report()

    st.sidebar.divider()
    st.sidebar.subheader("Diagnostic Centre")

    if not latest:
        st.sidebar.info(
            "No diagnostic report has been created yet."
        )
        return

    status = str(
        latest.get("status", "unknown")
    ).upper()

    if status == "SUCCESS":
        st.sidebar.success(
            f"Latest task: {status}"
        )
    else:
        st.sidebar.error(
            f"Latest task: {status}"
        )

    st.sidebar.write(
        f"**Stage:** {latest.get('stage', 'Unknown')}"
    )
    st.sidebar.write(
        f"**Script:** {latest.get('script_name', 'Unknown')}"
    )
    st.sidebar.write(
        f"**Duration:** "
        f"{latest.get('duration_seconds', 'Unknown')} seconds"
    )

    if latest.get("error_type"):
        st.sidebar.write(
            f"**Error:** {latest.get('error_type')}"
        )

    if latest.get("likely_cause"):
        st.sidebar.warning(
            latest.get("likely_cause")
        )

    if latest.get("suggested_fix"):
        st.sidebar.info(
            latest.get("suggested_fix")
        )

    location = []

    if latest.get("source_file"):
        location.append(
            str(latest.get("source_file"))
        )

    if latest.get("source_function"):
        location.append(
            str(latest.get("source_function"))
        )

    if latest.get("source_line"):
        location.append(
            f"line {latest.get('source_line')}"
        )

    if location:
        st.sidebar.caption(
            "Location: " + " • ".join(location)
        )

    report_json = json.dumps(
        latest,
        indent=2,
        ensure_ascii=False,
    )

    st.sidebar.download_button(
        "Download diagnostic report",
        data=report_json,
        file_name="podcastai_diagnostic.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.sidebar.expander(
        "Full diagnostic details"
    ):
        st.code(
            report_json,
            language="json",
        )



def display_home() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🎧 PodcastAI</div>
            <div class="hero-text">
                One upload. The right experience for every document.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF or image",
        type=["pdf", "jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        st.success(f"Selected: {uploaded_file.name}")

        if st.button("Analyse Material", type="primary"):
            with st.spinner("Reading and understanding your material..."):
                extraction_success, extraction_output = (
                    extract_uploaded_material(uploaded_file)
                )

                if extraction_success:
                    analysis_success, analysis_output = analyse_current_material()
                else:
                    analysis_success = False
                    analysis_output = ""

            if (
                extraction_success
                and analysis_success
                and read_text_file(EXTRACTED_TEXT_PATH)
            ):
                st.session_state.material_ready = True
                st.session_state.uploaded_name = uploaded_file.name
                st.success("Your material is ready.")
            else:
                st.error("The material could not be analysed.")

                details = "\n\n".join(
                    item
                    for item in (extraction_output, analysis_output)
                    if item
                )

                if details:
                    with st.expander("Technical details"):
                        st.code(details)

    if not st.session_state.material_ready:
        st.info("Upload and analyse a file to continue.")
        return

    display_analysis_card()

    st.divider()
    st.header("What would you like to create?")

    mode = current_mode()
    generation_mode = st.radio(
        "Generation mode",
        ["Quick Task", "Quick Pack", "Full Pack"],
        horizontal=True,
    )

    selected_tasks: list[str] = []

    if generation_mode == "Quick Task":
        selected_tasks = st.multiselect(
            "Choose one or more features",
            options=list(TASKS.keys()),
            default=["Summary"],
        )
    else:
        include_audio = st.checkbox(
            "Include story audio" if mode == "story" else "Include audio",
            value=False,
        )

        selected_tasks = tasks_for_mode(
            mode,
            generation_mode,
            include_audio,
        )

        st.caption("Included: " + ", ".join(selected_tasks))

    needs_audio = (
        "Podcast audio" in selected_tasks
        or "Podcast script" in selected_tasks
    )

    podcast_length = "Standard"

    if needs_audio:
        podcast_length = st.select_slider(
            "Audio length",
            options=["Quick", "Standard", "Deep Dive"],
            value="Standard",
            help=(
                "Quick is fastest and cheapest. Deep Dive takes "
                "longer and uses more API credit."
            ),
        )

    if st.button(
        "✨ Generate",
        type="primary",
        disabled=not selected_tasks,
    ):
        apply_mode_override(mode)

        cache_loaded = try_restore_cached_pack(
            podcast_length
        )

        if not cache_loaded:
            results = generate_tasks(
                selected_tasks,
                podcast_length,
            )

            st.session_state.generation_complete = any(
                success
                for success, _ in results.values()
            )

            successful_tasks = [
                task_name
                for task_name, (
                    success,
                    _,
                ) in results.items()
                if success
            ]

            if successful_tasks:
                save_current_pack_to_cache(
                    podcast_length=podcast_length,
                    generated_tasks=successful_tasks,
                )

                st.success(
                    "Results saved to the smart cache "
                    "for faster future loading."
                )

    if st.session_state.generation_complete:
        display_results()


def main() -> None:
    configure_page()
    initialise_state()
    display_developer_panel()
    display_home()


if __name__ == "__main__":
    main()