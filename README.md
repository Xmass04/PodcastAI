# PodcastAI V1 Beta Rollback Overlay

Cutoff used:
- 15 July 2026, 9:51 PM UK time (BST)
- Equivalent to 20:51 UTC

Purpose:
- Restore the last stable synchronous V1 Beta execution path.
- Remove background-worker, package-refactor, parallel Quick Pack and Audio V2
  experiments from the live execution path.
- Preserve diagnostics, adaptive modes, Story Mode, podcast lengths and smart cache.

Replace these live files:
- frontend/app.py
- backend/cache_manager.py
- backend/diagnostics.py
- backend/podcast_generator.py
- backend/podcast_audio.py

Keep the rest of the existing project unchanged:
- pdf_reader.py
- image_reader.py
- document_analyzer.py
- ai_summarizer.py
- study_notes.py
- flashcard_generator.py
- quiz_generator.py
- .streamlit/config.toml
- .env
- requirements files

Do not place files from v2_standby_do_not_import inside the live backend folder.
They are retained only for later controlled testing.

Start V1 with:
    streamlit run frontend/app.py

Before testing:
1. Stop Streamlit.
2. End leftover python/pythonw worker processes from later V2 trials.
3. Clear data/jobs if it exists.
4. Close podcast audio and podcast_script.txt previews.
5. Start Streamlit using the command above.

Recommended tests:
1. Small PDF -> Analyse Material.
2. Quick Task -> Summary.
3. Quick Pack without audio.
4. Story document -> Podcast script.
5. Study document -> Quick podcast audio.
6. Full Pack with audio.
7. Upload the same file again to test cache.
8. Enable Developer Mode and confirm diagnostics.
