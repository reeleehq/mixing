"""Guardrail: the base install (no ``media`` extra) keeps the pure modules importable.

The media backends (moviepy, opencv, scipy, pydub, burns, numpy, pillow, looks)
are only declared by the ``media`` extra. These tests hide them in a fresh
interpreter (``sys.modules[name] = None`` makes ``import name`` fail and
``find_spec`` return ``None``, exactly as on a machine without the wheel) and
check that

- the pure modules import and work, and load none of the hidden libraries;
- ``mixing.audio`` / ``mixing.video`` (and the lazy facade names that resolve to
  them) raise an ``ImportError`` naming ``mixing[media]``;
- the packaging metadata declares the split.
"""

import json
import subprocess
import sys
import textwrap

import pytest

from mixing.util import MEDIA_IMPORT_NAMES, MEDIA_INSTALL_HINT

_PROLOGUE = f"""
import sys
for _name in {list(MEDIA_IMPORT_NAMES)!r}:
    sys.modules[_name] = None
"""


def _run_hidden(body: str) -> dict:
    """Run ``body`` in a fresh interpreter with the media libs hidden; return its JSON."""
    code = _PROLOGUE + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_pure_modules_import_without_media_libs():
    out = _run_hidden(
        """
        import json, sys
        import mixing
        import mixing.srt, mixing.chapters, mixing.errors, mixing.util, mixing.egress
        import mixing.transcript, mixing.dubbing
        from mixing import Cue, parse_srt, dump_srt, MixingError, Chapter, has_ffmpeg
        from mixing.dubbing import text_to_speech, translate_srt
        from mixing import transcribe, remove_fillers
        cues = parse_srt("1\\n00:00:00,000 --> 00:00:01,500\\nhi\\n")
        print(json.dumps({
            "cues": len(cues),
            "loaded": [n for n in sys.modules
                       if n.split(".")[0] in ("moviepy", "cv2", "scipy", "pydub", "numpy", "PIL")
                       and sys.modules[n] is not None],
        }))
        """
    )
    assert out["cues"] == 1
    assert out["loaded"] == []


@pytest.mark.parametrize(
    "statement",
    [
        "import mixing.audio",
        "import mixing.video",
        "from mixing.video import replace_audio",
        "from mixing.audio import fade_in",
        "import mixing.video.video_ops",
        "import mixing.audio.segmentation",
        "import mixing; mixing.Video",
        "import mixing; mixing.Audio",
        "from mixing import dub_video_from_srt",
    ],
)
def test_media_modules_raise_import_error_naming_the_extra(statement):
    out = _run_hidden(
        f"""
        import json
        try:
            {statement}
        except ImportError as e:
            print(json.dumps({{"message": str(e)}}))
        else:
            print(json.dumps({{"message": None}}))
        """
    )
    assert out["message"] is not None, "expected an ImportError"
    assert "pip install 'mixing[media]'" in out["message"]
    assert MEDIA_INSTALL_HINT == "pip install 'mixing[media]'"


def test_require_package_hint_names_the_extra_for_media_libs():
    out = _run_hidden(
        """
        import json
        from mixing.util import require_package
        try:
            require_package("scipy.signal")
        except ImportError as e:
            print(json.dumps({"message": str(e)}))
        """
    )
    assert "mixing[media]" in out["message"]


def test_require_package_hint_unchanged_for_other_packages():
    from mixing.util import require_package

    with pytest.raises(ImportError, match="pip install nonexistent_package_xyz123"):
        require_package("nonexistent_package_xyz123")


def test_pyproject_declares_the_media_extra():
    tomllib = pytest.importorskip("tomllib" if sys.version_info >= (3, 11) else "tomli")
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not path.exists():
        pytest.skip("pyproject.toml not available (installed distribution)")
    project = tomllib.loads(path.read_text())["project"]

    def names(requirements):
        import re

        return {re.split(r"[<>=!~\[ ;]", r, maxsplit=1)[0].lower() for r in requirements}

    heavy = {"moviepy", "opencv-contrib-python", "scipy", "pydub", "burns", "numpy"}
    assert not heavy & names(project["dependencies"])
    assert heavy <= names(project["optional-dependencies"]["media"])
