from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_FOLDER = PROJECT_ROOT / "backend"
DATA_FOLDER = PROJECT_ROOT / "data"

EXTRACTED_TEXT_PATH = (
    DATA_FOLDER / "extracted_text" / "output.txt"
)

TASK_SCRIPTS = {
    "analysis": "document_analyzer.py",
    "summary": "ai_summarizer.py",
    "notes": "study_notes.py",
    "flashcards": "flashcard_generator.py",
    "quiz": "quiz_generator.py",
    "podcast_script": "podcast_generator.py",
    "podcast_audio": "podcast_audio.py",
}


def display_header() -> None:
    print("\n" + "=" * 58)
    print("                    PodcastAI")
    print("       One upload. Everything you need to learn.")
    print("=" * 58)


def run_script(
    script_name: str,
    input_text: str | None = None,
) -> bool:
    """Run one PodcastAI backend module."""

    script_path = BACKEND_FOLDER / script_name

    if not script_path.exists():
        print(f"\n❌ Missing script: {script_path}")
        return False

    print(f"\n▶ Running {script_name}...")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            input=input_text,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print(
                f"\n❌ {script_name} stopped with "
                f"exit code {result.returncode}."
            )
            return False

        print(f"✅ Finished {script_name}")
        return True

    except OSError as error:
        print(f"\n❌ Could not run {script_name}: {error}")
        return False


def extract_from_pdf() -> bool:
    """Run the existing PDF reader using a user-selected file."""

    raw_path = input(
        "\nEnter or paste the path to your PDF:\n> "
    ).strip().strip('"')

    pdf_path = Path(raw_path)

    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    if not pdf_path.exists():
        print(f"\n❌ File not found: {pdf_path}")
        return False

    if pdf_path.suffix.lower() != ".pdf":
        print("\n❌ The selected file is not a PDF.")
        return False

    # pdf_reader.py expects the path through input().
    return run_script(
        "pdf_reader.py",
        input_text=f"{pdf_path}\n",
    )


def extract_from_images() -> bool:
    """
    Run the image reader.

    The existing image reader processes supported images stored in
    data/images in alphabetical order.
    """

    image_folder = DATA_FOLDER / "images"
    image_folder.mkdir(parents=True, exist_ok=True)

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    image_files = sorted(
        path
        for path in image_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )

    if not image_files:
        print(
            "\n❌ No images were found in:\n"
            f"{image_folder}\n\n"
            "Add JPG, JPEG, PNG or WEBP files, then try again."
        )
        return False

    print(f"\nFound {len(image_files)} image page(s):")

    for image_file in image_files:
        print(f"• {image_file.name}")

    return run_script("image_reader.py")


def choose_input() -> bool:
    """Ask how the user wants to provide source material."""

    print("\nChoose your source material:")
    print("1. PDF")
    print("2. Images")
    print("3. Use the text already extracted")

    choice = input("\nSelect 1, 2 or 3:\n> ").strip()

    if choice == "1":
        return extract_from_pdf()

    if choice == "2":
        return extract_from_images()

    if choice == "3":
        if not EXTRACTED_TEXT_PATH.exists():
            print("\n❌ No previously extracted text was found.")
            return False

        if not EXTRACTED_TEXT_PATH.read_text(
            encoding="utf-8"
        ).strip():
            print("\n❌ The extracted-text file is empty.")
            return False

        print("\n✅ Existing extracted text selected.")
        return True

    print("\n❌ Invalid input selection.")
    return False


def run_task(task_name: str) -> bool:
    """Run a named generation task."""

    script_name = TASK_SCRIPTS[task_name]
    return run_script(script_name)


def generate_quick_pack() -> None:
    """Generate the fastest useful learning resources."""

    print("\n⚡ Creating your Quick Pack...")

    tasks = [
        ("analysis", "Document analysis"),
        ("summary", "Quick summary"),
        ("flashcards", "Flashcards"),
        ("quiz", "Quiz"),
    ]

    completed = 0

    for task_name, display_name in tasks:
        print(f"\nCreating: {display_name}")

        if run_task(task_name):
            completed += 1
        else:
            print(
                f"⚠ {display_name} could not be completed. "
                "Continuing with the remaining tasks."
            )

    print(
        f"\n⚡ Quick Pack complete: "
        f"{completed}/{len(tasks)} tasks succeeded."
    )


def generate_full_pack(
    include_audio: bool = True,
) -> None:
    """Generate the complete PodcastAI content pack."""

    print("\n📚 Creating your Full Study Pack...")

    tasks = [
        ("analysis", "Document analysis"),
        ("summary", "AI summary"),
        ("notes", "Structured study notes"),
        ("flashcards", "Flashcards"),
        ("quiz", "Quiz"),
        ("podcast_script", "Podcast script"),
    ]

    if include_audio:
        tasks.append(
            ("podcast_audio", "Two-voice podcast audio")
        )

    completed = 0

    for task_name, display_name in tasks:
        print("\n" + "-" * 58)
        print(f"Creating: {display_name}")

        if run_task(task_name):
            completed += 1
        else:
            print(
                f"⚠ {display_name} could not be completed. "
                "The pipeline will continue."
            )

    print("\n" + "=" * 58)
    print(
        f"📚 Full Pack finished: "
        f"{completed}/{len(tasks)} tasks succeeded."
    )
    print("=" * 58)


def choose_individual_task() -> None:
    """Allow one feature to be generated independently."""

    options = {
        "1": ("analysis", "Document analysis"),
        "2": ("summary", "AI summary"),
        "3": ("notes", "Study notes"),
        "4": ("flashcards", "Flashcards"),
        "5": ("quiz", "Quiz"),
        "6": ("podcast_script", "Podcast script"),
        "7": ("podcast_audio", "Podcast audio"),
    }

    print("\nChoose a Quick Task:")

    for number, (_, display_name) in options.items():
        print(f"{number}. {display_name}")

    choice = input("\nSelect a task:\n> ").strip()

    selected = options.get(choice)

    if selected is None:
        print("\n❌ Invalid task selection.")
        return

    task_name, display_name = selected

    print(f"\nCreating: {display_name}")

    if run_task(task_name):
        print(f"\n✅ {display_name} is ready.")
    else:
        print(f"\n❌ {display_name} could not be created.")


def choose_generation_mode() -> None:
    """Choose between individual, quick and complete generation."""

    print("\nWhat would you like to create?")
    print("1. Quick Task — generate one selected item")
    print("2. Quick Pack — summary, cards and quiz")
    print("3. Full Study Pack — generate everything")

    choice = input("\nSelect 1, 2 or 3:\n> ").strip()

    if choice == "1":
        choose_individual_task()
        return

    if choice == "2":
        generate_quick_pack()
        return

    if choice == "3":
        audio_choice = input(
            "\nInclude podcast audio?\n"
            "This may take longer and use more API credit.\n"
            "Enter Y or N:\n> "
        ).strip().lower()

        generate_full_pack(
            include_audio=audio_choice in {"y", "yes"},
        )
        return

    print("\n❌ Invalid generation mode.")


def main() -> None:
    display_header()

    if not choose_input():
        print("\nPipeline stopped because extraction failed.")
        return

    if not EXTRACTED_TEXT_PATH.exists():
        print("\n❌ No extracted text was produced.")
        return

    extracted_text = EXTRACTED_TEXT_PATH.read_text(
        encoding="utf-8"
    )

    if not extracted_text.strip():
        print("\n❌ The extracted text is empty.")
        return

    print("\n✅ Your material is ready for processing.")

    choose_generation_mode()

    print(
        "\nPodcastAI has finished processing your request."
    )


if __name__ == "__main__":
    main()