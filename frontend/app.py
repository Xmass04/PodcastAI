from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import streamlit as st


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
PODCAST_AUDIO_PATH = (
    DATA_FOLDER / "podcasts" / "audio" / "podcast_episode.wav"
)

TASKS = {
    "Document analysis": "document_analyzer.py",
    "Summary": "ai_summarizer.py",
    "Study notes": "study_notes.py",
    "Flashcards": "flashcard_generator.py",
    "Quiz": "quiz_generator.py",
    "Podcast script": "podcast_generator.py",
    "Podcast audio": "podcast_audio.py",
}


def configure_page() -> None:
    st.set_page_config(
        page_title="PodcastAI",
        page_icon="🎓",
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

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        .hero {
            text-align: center;
            padding: 2.2rem 1rem 1.5rem;
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 750;
            margin-bottom: 0.4rem;
        }

        .hero-text {
            font-size: 1.15rem;
            color: #5b6472;
            margin-bottom: 1.5rem;
        }

        .info-card {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin: 0.75rem 0;
        }

        .small-note {
            color: #697386;
            font-size: 0.9rem;
        }

        div[data-testid="stFileUploader"] {
            border-radius: 16px;
        }

        div.stButton > button {
            width: 100%;
            min-height: 3rem;
            border-radius: 12px;
            font-weight: 650;
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
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def run_backend_script(
    script_name: str,
    input_text: str | None = None,
) -> tuple[bool, str]:
    script_path = BACKEND_FOLDER / script_name

    if not script_path.exists():
        return False, f"Missing backend file: {script_path}"

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
        )

    except OSError as error:
        return False, str(error)

    combined_output = "\n".join(
        part for part in (result.stdout, result.stderr) if part
    )

    return result.returncode == 0, combined_output.strip()


def clear_folder(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    for item in folder.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def save_uploaded_file(uploaded_file) -> Path:
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    safe_name = Path(uploaded_file.name).name
    destination = UPLOAD_FOLDER / safe_name
    destination.write_bytes(uploaded_file.getbuffer())

    return destination


def extract_uploaded_material(uploaded_file) -> tuple[bool, str]:
    uploaded_path = save_uploaded_file(uploaded_file)
    extension = uploaded_path.suffix.lower()

    if extension == ".pdf":
        return run_backend_script(
            "pdf_reader.py",
            input_text=f"{uploaded_path}\n",
        )

    if extension in {".jpg", ".jpeg", ".png", ".webp"}:
        clear_folder(IMAGE_FOLDER)

        image_destination = IMAGE_FOLDER / uploaded_path.name
        shutil.copy2(uploaded_path, image_destination)

        return run_backend_script("image_reader.py")

    return False, "Unsupported file type."


def generate_tasks(
    selected_tasks: list[str],
) -> dict[str, tuple[bool, str]]:
    results: dict[str, tuple[bool, str]] = {}

    progress = st.progress(
        0,
        text="Preparing your content...",
    )

    with st.status(
        "Creating your PodcastAI pack...",
        expanded=True,
    ) as status:
        total = len(selected_tasks)

        for index, task_name in enumerate(
            selected_tasks,
            start=1,
        ):
            st.write(f"Creating **{task_name}**...")

            script_name = TASKS[task_name]
            success, output = run_backend_script(script_name)
            results[task_name] = (success, output)

            if success:
                st.write(f"✅ {task_name} ready")
            else:
                st.write(f"⚠️ {task_name} could not be completed")

            percentage = int((index / total) * 100)

            progress.progress(
                percentage,
                text=f"{task_name}: {'ready' if success else 'failed'}",
            )

        successful = sum(
            1 for success, _ in results.values() if success
        )

        if successful == total:
            status.update(
                label="Your content is ready!",
                state="complete",
                expanded=False,
            )
        else:
            status.update(
                label=(
                    f"Finished with {successful}/{total} "
                    "items successfully created"
                ),
                state="complete",
                expanded=True,
            )

    return results


def read_text_file(path: Path) -> str | None:
    if not path.exists():
        return None

    content = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    return content or None


def display_text_result(
    title: str,
    path: Path,
    download_name: str,
) -> None:
    content = read_text_file(path)

    st.subheader(title)

    if not content:
        st.info(f"{title} has not been generated yet.")
        return

    st.markdown(content)

    st.download_button(
        label=f"Download {title}",
        data=content,
        file_name=download_name,
        mime="text/plain",
        use_container_width=True,
    )


def display_json_result(
    title: str,
    path: Path,
    download_name: str,
) -> None:
    content = read_text_file(path)

    st.subheader(title)

    if not content:
        st.info(f"{title} has not been generated yet.")
        return

    st.code(content, language="json")

    st.download_button(
        label=f"Download {title}",
        data=content,
        file_name=download_name,
        mime="application/json",
        use_container_width=True,
    )


def display_results() -> None:
    st.divider()
    st.header("Your results")

    tabs = st.tabs(
        [
            "Overview",
            "Study Notes",
            "Flashcards",
            "Quiz",
            "Podcast",
        ]
    )

    with tabs[0]:
        analysis = read_text_file(ANALYSIS_PATH)

        if analysis:
            with st.expander(
                "Document analysis",
                expanded=False,
            ):
                st.code(analysis, language="json")

        display_text_result(
            "Summary",
            SUMMARY_PATH,
            "summary.txt",
        )

    with tabs[1]:
        display_text_result(
            "Study Notes",
            NOTES_PATH,
            "study_notes.txt",
        )

    with tabs[2]:
        display_json_result(
            "Flashcards",
            FLASHCARDS_PATH,
            "flashcards.json",
        )

    with tabs[3]:
        display_json_result(
            "Quiz",
            QUIZ_PATH,
            "quiz.json",
        )

    with tabs[4]:
        script = read_text_file(PODCAST_SCRIPT_PATH)

        if PODCAST_AUDIO_PATH.exists():
            st.subheader("Podcast episode")
            st.audio(str(PODCAST_AUDIO_PATH))

            st.caption(
                "This episode uses AI-generated voices."
            )

            st.download_button(
                label="Download Podcast",
                data=PODCAST_AUDIO_PATH.read_bytes(),
                file_name="podcast_episode.wav",
                mime="audio/wav",
                use_container_width=True,
            )
        else:
            st.info("Podcast audio has not been generated yet.")

        if script:
            with st.expander("Read podcast script"):
                st.text(script)


def display_home() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🎓 PodcastAI</div>
            <div class="hero-text">
                One upload. Start learning your way.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <strong>Upload your material</strong><br>
            <span class="small-note">
                Use a PDF or a clear image of printed notes.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF or image",
        type=["pdf", "jpg", "jpeg", "png", "webp"],
        help=(
            "Uploaded material remains local during this "
            "development version."
        ),
    )

    if uploaded_file is not None:
        st.success(f"Selected: {uploaded_file.name}")

        if st.button(
            "Analyse Material",
            type="primary",
        ):
            with st.spinner("Reading your material..."):
                success, output = extract_uploaded_material(
                    uploaded_file
                )

            if success and EXTRACTED_TEXT_PATH.exists():
                extracted = read_text_file(
                    EXTRACTED_TEXT_PATH
                )

                if extracted:
                    st.session_state.material_ready = True
                    st.session_state.uploaded_name = (
                        uploaded_file.name
                    )
                    st.success(
                        "Your material is ready for generation."
                    )
                else:
                    st.error(
                        "No readable text was extracted."
                    )
            else:
                st.error(
                    "The material could not be processed."
                )

                if output:
                    with st.expander("Technical details"):
                        st.code(output)

    if not st.session_state.material_ready:
        st.info(
            "Upload and analyse a file to unlock generation options."
        )
        return

    st.divider()

    st.header("What would you like to create?")

    mode = st.radio(
        "Generation mode",
        options=[
            "Quick Task",
            "Quick Pack",
            "Full Study Pack",
        ],
        horizontal=True,
    )

    selected_tasks: list[str]

    if mode == "Quick Task":
        selected_tasks = st.multiselect(
            "Choose one or more features",
            options=list(TASKS.keys()),
            default=["Summary"],
        )

    elif mode == "Quick Pack":
        st.caption(
            "A fast overview with a summary, flashcards and quiz."
        )

        selected_tasks = [
            "Document analysis",
            "Summary",
            "Flashcards",
            "Quiz",
        ]

    else:
        include_audio = st.checkbox(
            "Include two-voice podcast audio",
            value=False,
            help=(
                "Audio takes longer and may use more API credit."
            ),
        )

        selected_tasks = [
            "Document analysis",
            "Summary",
            "Study notes",
            "Flashcards",
            "Quiz",
            "Podcast script",
        ]

        if include_audio:
            selected_tasks.append("Podcast audio")

    if st.button(
        "✨ Generate",
        type="primary",
        disabled=not selected_tasks,
    ):
        results = generate_tasks(selected_tasks)

        st.session_state.generation_complete = any(
            success for success, _ in results.values()
        )

        failed_outputs = {
            task: output
            for task, (success, output) in results.items()
            if not success
        }

        if failed_outputs:
            with st.expander("Generation details"):
                for task, output in failed_outputs.items():
                    st.write(f"**{task}**")
                    st.code(output or "No details returned.")

    if st.session_state.generation_complete:
        display_results()


def main() -> None:
    configure_page()
    initialise_state()
    display_home()


if __name__ == "__main__":
    main()