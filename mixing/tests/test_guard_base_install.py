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

import re
import json
import subprocess
import sys
import textwrap

import pytest

from mixing.util import MEDIA_IMPORT_NAMES, MEDIA_INSTALL_HINT

_PROLOGUE = """
import sys
for _name in {hidden!r}:
    sys.modules[_name] = None
"""


def _run_hidden(body: str, *, hidden=MEDIA_IMPORT_NAMES) -> dict:
    """Run ``body`` in a fresh interpreter with ``hidden`` modules (default: the
    media libs) made unimportable; return its JSON."""
    code = _PROLOGUE.format(hidden=list(hidden)) + textwrap.dedent(body)
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


def test_require_package_error_names_the_requested_package():
    # callers (foley's CLI) map ``exc.name`` to their own install hint
    from mixing.util import require_package

    with pytest.raises(ImportError) as info:
        require_package("nonexistent_package_xyz123.sub")
    assert info.value.name == "nonexistent_package_xyz123.sub"


def test_require_extra_error_names_the_first_missing_module():
    from mixing.util import require_ml_extra

    with pytest.raises(ImportError) as info:
        require_ml_extra("demo", modules=("math", "no_such_module_xyz"))
    assert info.value.name == "no_such_module_xyz"


def test_pyproject_declares_the_media_extra():
    tomllib = pytest.importorskip("tomllib" if sys.version_info >= (3, 11) else "tomli")
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not path.exists():
        pytest.skip("pyproject.toml not available (installed distribution)")
    project = tomllib.loads(path.read_text())["project"]

    def names(requirements):
        import re

        return {
            re.split(r"[<>=!~\[ ;]", r, maxsplit=1)[0].lower() for r in requirements
        }

    heavy = {"moviepy", "opencv-contrib-python", "scipy", "pydub", "burns", "numpy"}
    assert not heavy & names(project["dependencies"])
    assert heavy <= names(project["optional-dependencies"]["media"])


def test_loudness_without_numpy_names_the_missing_module():
    out = _run_hidden(
        """
        import json
        try:
            import mixing.loudness
        except ImportError as e:
            print(json.dumps({"name": e.name, "message": str(e)}))
        """
    )
    assert out["name"] == "numpy"
    assert "mixing[loudness]" in out["message"]


#: Import name -> distribution name, where they differ. IPython's own
#: distribution is ``ipython``; the ``widget`` extra gets it from the ``jupyter``
#: metapackage it declares (which has no import name), so IPython stands for it.
_DISTRIBUTIONS = {
    "cv2": "opencv-contrib-python",
    "PIL": "pillow",
    "google.genai": "google-genai",
    "IPython": "jupyter",
}
#: Extras that install tools and other extras, not modules mixing imports.
_NOT_IN_TABLE = {"dev", "testing"}


def _declared_extras() -> dict:
    """``{extra: {distribution name}}`` from pyproject's optional-dependencies."""
    tomllib = pytest.importorskip("tomllib" if sys.version_info >= (3, 11) else "tomli")
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not path.exists():
        pytest.skip("pyproject.toml not available (installed distribution)")
    declared = tomllib.loads(path.read_text())["project"]["optional-dependencies"]

    from packaging.requirements import Requirement

    def distributions(extra, seen=()):
        """``extra``'s distributions, with ``mixing[x]`` read as extra ``x``'s."""
        assert extra not in seen, f"extras include each other: {seen + (extra,)}"
        names = set()
        for requirement in map(Requirement, declared[extra]):
            if requirement.name.lower() == "mixing":
                for inner in requirement.extras:
                    names |= distributions(inner, seen + (extra,))
            else:
                names.add(requirement.name.lower())
        return names

    return {extra: distributions(extra) for extra in declared}


def test_the_table_and_pyproject_declare_the_same_extras():
    """``EXTRAS`` is pyproject's extras read as import names: the same extras,
    each with the same packages (both directions, so neither can drift)."""
    from mixing.util import EXTRAS

    declared = _declared_extras()
    assert set(EXTRAS) == set(declared) - _NOT_IN_TABLE
    for extra, modules in EXTRAS.items():
        supplied = {_DISTRIBUTIONS.get(m, m).lower() for m in modules}
        assert supplied == declared[extra], (
            f"EXTRAS[{extra!r}] supplies {sorted(supplied)}, "
            f"pyproject's {extra!r} declares {sorted(declared[extra])}"
        )


@pytest.mark.parametrize(
    "module, hint",
    [
        ("scipy.signal", "pip install 'mixing[media]'"),
        ("torch", "pip install 'mixing[ml]'"),
        ("librosa", "pip install 'mixing[beats]'"),
        ("numpy", "pip install 'mixing[media]'"),
        ("aix", "pip install 'mixing[llm]'"),
        ("google.genai.types", "pip install 'mixing[gen]'"),
        ("pyclip", "pip install 'mixing[clipboard]'"),
    ],
)
def test_require_package_hints_the_extra_from_the_table(monkeypatch, module, hint):
    from mixing.util import require_package

    _deny_imports(monkeypatch)
    with pytest.raises(ImportError, match=re.escape(hint)):
        require_package(module)


def test_unknown_extra_and_empty_module_list():
    from mixing.util import install_hint, require_extra

    with pytest.raises(ValueError, match="unknown extra"):
        install_hint("nope")
    require_extra("demo", "ml", modules=())  # nothing to check: no error


def _deny_imports(monkeypatch):
    import importlib

    def deny(name, *a, **k):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", deny)


def test_require_package_names_the_callers_extra(monkeypatch):
    from mixing.util import require_package

    _deny_imports(monkeypatch)
    with pytest.raises(ImportError, match=re.escape("mixing[loudness]")) as info:
        require_package("scipy.signal", extra="loudness")
    assert info.value.name == "scipy.signal"


def test_require_package_rejects_an_unknown_extra_even_when_installed():
    from mixing.util import require_package

    with pytest.raises(ValueError, match="unknown extra"):
        require_package("math", extra="nope")


def test_loudness_without_scipy_names_the_loudness_extra():
    """numpy installed, scipy not: the meter names ``mixing[loudness]`` (the
    extra the user is on), not ``media``, and the ffmpeg meter's module loads."""
    out = _run_hidden(
        """
        import json
        import numpy as np
        import mixing.loudness as L
        x = np.sin(np.arange(8000) / 10.0)[:, None] * 0.1
        ffmpeg = L.integrated_lufs(x, 8000, meter="ffmpeg_ebur128")
        try:
            L.integrated_lufs(x, 8000)
        except ImportError as e:
            print(json.dumps({"name": e.name, "message": str(e), "ffmpeg": ffmpeg}))
        """,
        hidden=("scipy",),
    )
    assert out["name"] == "scipy.signal"
    assert "pip install 'mixing[loudness]'" in out["message"]
    assert isinstance(out["ffmpeg"], float)


def test_llm_defaults_name_the_llm_extra_and_the_alternative():
    out = _run_hidden(
        """
        import json
        from mixing.chapters import default_segment_fn
        from mixing.dubbing.srt import default_translate_fn
        msgs = []
        for call in (lambda: default_segment_fn([], 1),
                     lambda: default_translate_fn(["hi"], "fr")):
            try:
                call()
            except ImportError as e:
                msgs.append({"name": e.name, "message": str(e)})
        print(json.dumps(msgs))
        """,
        hidden=("aix",),
    )
    assert [m["name"] for m in out] == ["aix", "aix"]
    assert all("pip install 'mixing[llm]'" in m["message"] for m in out)
    assert "segment_fn to detect_chapters()" in out[0]["message"]
    assert "translate_fn to translate_srt()" in out[1]["message"]


def test_audio_widget_without_soundfile_names_the_widget_extra():
    out = _run_hidden(
        """
        import json, sys, types
        for _stub in ("IPython", "IPython.display"):  # Jupyter is not installed in CI
            sys.modules.setdefault(_stub, types.ModuleType(_stub))
        sys.modules["IPython.display"].display = sys.modules["IPython.display"].HTML = None
        from mixing.audio.audio_widget import ensure_wfsr
        try:
            ensure_wfsr("x.wav")
        except ImportError as e:
            print(json.dumps({"name": e.name, "message": str(e)}))
        """,
        hidden=("soundfile",),
    )
    assert out["name"] == "soundfile"
    assert "pip install 'mixing[widget]'" in out["message"]


def test_audio_widget_without_jupyter_names_the_widget_extra():
    out = _run_hidden(
        """
        import json
        try:
            import mixing.audio.audio_widget
        except ImportError as e:
            print(json.dumps({"name": e.name, "message": str(e)}))
        """,
        hidden=("IPython",),
    )
    assert out["name"].split(".")[0] == "IPython"
    assert "pip install 'mixing[widget]'" in out["message"]


def test_the_audio_extra_is_an_alias_of_media_plus_soundfile():
    """Downstream installs name ``mixing[audio]`` (paces): it stays installable
    and supplies the media backends, but no module hints it for them."""
    from mixing.util import EXTRAS, extra_for

    assert _declared_extras()["audio"] == _declared_extras()["media"] | {"soundfile"}
    assert set(EXTRAS["audio"]) == set(EXTRAS["media"]) | {"soundfile"}
    assert all(extra_for(m) != "audio" for m in EXTRAS["audio"])
    assert extra_for("soundfile") == "widget"


def test_require_package_does_not_blame_an_installed_package(monkeypatch):
    """A package that is there but whose own import fails re-raises that error."""
    import importlib

    from mixing.util import require_package

    def broken(name, *a, **k):
        raise ModuleNotFoundError("No module named 'numba'", name="numba")

    monkeypatch.setattr(importlib, "import_module", broken)
    with pytest.raises(ModuleNotFoundError) as info:
        require_package("librosa")
    assert info.value.name == "numba"
    assert "mixing[" not in str(info.value)
