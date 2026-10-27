"""General utilities for mixing video, audio, etc."""

from typing import Literal, Optional
import importlib
import importlib.util
import logging
import shutil

logger = logging.getLogger(__name__)

TimeUnit = Literal["seconds", "frames", "milliseconds"]
AudioTimeUnit = Literal["seconds", "samples", "milliseconds"]


def has_ffmpeg() -> bool:
    """Return True if the ``ffmpeg`` binary is available on ``PATH``.

    Most real video work in ``mixing`` (concatenation, trim/pad, dimension
    normalization, audio replacement) shells out to ffmpeg via moviepy.
    Without it, some operations degrade silently — producing wrong-duration
    or lower-quality output instead of failing. Callers should check this at
    startup and refuse to run rather than emit broken media.

    >>> isinstance(has_ffmpeg(), bool)
    True
    """
    return shutil.which("ffmpeg") is not None


def ffmpeg_exe() -> str:
    """Resolve an ffmpeg executable, preferring the pip-bundled binary.

    Resolution order is ``imageio-ffmpeg`` (which itself honors the
    ``IMAGEIO_FFMPEG_EXE`` env var, then its bundled static binary, then the
    system ``PATH``) and finally a bare ``PATH`` lookup. This is what lets
    subprocess-ffmpeg features work on a machine where only ``pip install``
    ran — ``imageio-ffmpeg`` is a moviepy dependency, so it is present
    wherever mixing's video stack is.

    Raises:
        RuntimeError: when no ffmpeg executable can be found, naming both
            remedies (installing ``imageio-ffmpeg`` or a system ffmpeg).

    >>> isinstance(ffmpeg_exe(), str)  # doctest: +SKIP
    True
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        raise RuntimeError(
            "No ffmpeg executable found. Install the pip-bundled binary "
            "(`pip install imageio-ffmpeg`) or a system ffmpeg "
            "(https://ffmpeg.org/) and ensure it is on PATH."
        )


MEDIA_EXTRA = "media"
ML_EXTRA = "ml"
BEATS_EXTRA = "beats"
LOUDNESS_EXTRA = "loudness"
AUDIO_EXTRA = "audio"
WIDGET_EXTRA = "widget"
GEN_EXTRA = "gen"
LLM_EXTRA = "llm"
CLIPBOARD_EXTRA = "clipboard"

# Import names (top-level) that the ``media`` extra provides, per media area.
AUDIO_IMPORT_NAMES = ("numpy", "pydub", "scipy")
VIDEO_IMPORT_NAMES = ("numpy", "moviepy", "cv2", "PIL", "burns", "looks")
MEDIA_IMPORT_NAMES = tuple(dict.fromkeys(AUDIO_IMPORT_NAMES + VIDEO_IMPORT_NAMES))
# ``ml``: pretrained models (the AudioSet tagger behind the "audioset"
# segmentation strategy), imported lazily inside the functions that need them.
ML_IMPORT_NAMES = ("torch", "transformers")
BEATS_IMPORT_NAMES = ("librosa",)
LOUDNESS_IMPORT_NAMES = ("numpy", "scipy")

#: The one table of mixing's extras: ``{extra: import names it supplies}``,
#: each declared by the same-named extra in ``pyproject.toml`` (a test holds the
#: two equal). A name may be dotted (``google.genai``: ``google`` is a shared
#: namespace). A module in several extras (numpy) maps to the first, so the
#: broadest install wins; a caller that knows better names its extra
#: (``require_package("scipy.signal", extra=LOUDNESS_EXTRA)``). ``extra_for`` /
#: ``install_hint`` are public so a caller can map a missing module
#: (``ImportError.name``) to its install line.
EXTRAS: dict[str, tuple[str, ...]] = {
    MEDIA_EXTRA: MEDIA_IMPORT_NAMES,
    ML_EXTRA: ML_IMPORT_NAMES,
    BEATS_EXTRA: BEATS_IMPORT_NAMES,
    LOUDNESS_EXTRA: LOUDNESS_IMPORT_NAMES,
    AUDIO_EXTRA: ("pydub", "ffmpeg", "soundfile"),
    WIDGET_EXTRA: ("ipywidgets", "soundfile", "IPython"),
    GEN_EXTRA: ("google.genai",),
    LLM_EXTRA: ("aix",),
    CLIPBOARD_EXTRA: ("pyclip",),
}


def install_hint(extra: str) -> str:
    """The pip line that installs ``extra``.

    >>> install_hint("media")
    "pip install 'mixing[media]'"
    """
    if extra not in EXTRAS:
        raise ValueError(f"unknown extra {extra!r}; one of {sorted(EXTRAS)}")
    return f"pip install 'mixing[{extra}]'"


def _supplies(name: str, module: str) -> bool:
    """True if import name ``name`` is ``module`` or one of its parent packages."""
    return module == name or module.startswith(name + ".")


def extra_for(module: str) -> Optional[str]:
    """The extra that supplies ``module`` (or a parent package of it), or ``None``.

    >>> extra_for("scipy.signal"), extra_for("librosa"), extra_for("google.genai.types")
    ('media', 'beats', 'gen')
    >>> extra_for("requests"), extra_for("google.protobuf")
    (None, None)
    """
    return next(
        (
            extra
            for extra, names in EXTRAS.items()
            if any(_supplies(name, module) for name in names)
        ),
        None,
    )


MEDIA_INSTALL_HINT = install_hint(MEDIA_EXTRA)
ML_INSTALL_HINT = install_hint(ML_EXTRA)
BEATS_INSTALL_HINT = install_hint(BEATS_EXTRA)


def _is_importable(name: str) -> bool:
    """True if top-level module ``name`` can be found, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def require_extra(
    feature: str, extra: str, *, modules: Optional[tuple[str, ...]] = None
) -> None:
    """Raise an informative ``ImportError`` if ``extra`` is not installed.

    Checks ``modules`` (default: every module the extra supplies) without
    importing them; the error names the extra's install line, and its
    ``name`` is the first missing module.

    >>> require_extra("demo", "ml", modules=("math",))
    >>> require_extra("demo", "ml", modules=("no_such_module_xyz",))
    Traceback (most recent call last):
      ...
    ImportError: demo requires the 'ml' extra ...
    """
    hint = install_hint(extra)
    modules = EXTRAS[extra] if modules is None else modules
    missing = [name for name in modules if not _is_importable(name)]
    if missing:
        raise ImportError(
            f"{feature} requires the {extra!r} extra "
            f"(missing: {', '.join(missing)}). Install it with {hint}.",
            name=missing[0],
        )


def require_media_extra(feature: str, *, modules: tuple[str, ...] = MEDIA_IMPORT_NAMES):
    """:func:`require_extra` for ``media``: the moviepy / opencv / scipy / pydub
    backends that the base ``pip install mixing`` (pure modules only) lacks.

    >>> require_media_extra("demo", modules=("no_such_module_xyz",))
    Traceback (most recent call last):
      ...
    ImportError: demo requires the 'media' extra ...
    """
    require_extra(feature, MEDIA_EXTRA, modules=modules)


def require_ml_extra(feature: str, *, modules: tuple[str, ...] = ML_IMPORT_NAMES):
    """:func:`require_extra` for ``ml`` (``torch`` / ``transformers``)."""
    require_extra(feature, ML_EXTRA, modules=modules)


def require_package(package_name: str, *, extra: Optional[str] = None):
    """
    Import a package, raising an informative error if not installed.

    The hint names ``extra`` when given (the extra the caller belongs to: in
    ``mixing.loudness``, scipy comes from ``loudness``, not ``media``), else the
    extra ``EXTRAS`` maps ``package_name`` to, else ``pip install <package>``.
    The error's ``name`` is ``package_name``, so a caller can map it to its own
    install hint.

    >>> math = require_package('math')  # doctest: +SKIP
    >>> math.pi  # doctest: +SKIP
    3.141592653589793
    """
    if extra is not None:
        install_hint(extra)  # an unknown extra is the caller's bug: fail now
    try:
        return importlib.import_module(package_name)
    except ImportError as e:
        extra = extra or extra_for(package_name)
        hint = install_hint(extra) if extra else f"pip install {package_name}"
        raise ImportError(
            f"Package '{package_name}' is required for this functionality. "
            f"Please install it via '{hint}'.",
            name=package_name,
        ) from e


def require_llm_chat(feature: str, *, alternative: str):
    """``aix.chat``, the LLM call behind ``feature``, or an ``ImportError`` naming
    the ``llm`` extra and the ``alternative`` that needs no LLM.

    >>> require_llm_chat("demo", alternative="segment_fn")  # doctest: +SKIP
    """
    try:
        return require_package("aix", extra=LLM_EXTRA).chat
    except ImportError as e:
        raise ImportError(
            f"{feature} needs an LLM. {e} Or pass your own {alternative}.",
            name=e.name,
        ) from e


def to_seconds(value: float, *, unit: TimeUnit | AudioTimeUnit, rate: float) -> float:
    """
    Convert time value to seconds based on unit.

    Args:
        value: Time value to convert
        unit: Unit of the value ('seconds', 'frames', 'milliseconds', 'samples')
        rate: Frame rate (fps) for 'frames' or sample rate (Hz) for 'samples'

    Returns:
        Time in seconds

    Examples:
        >>> to_seconds(10, unit="seconds", rate=24)  # doctest: +SKIP
        10.0
        >>> to_seconds(240, unit="frames", rate=24)  # doctest: +SKIP
        10.0
        >>> to_seconds(10000, unit="milliseconds", rate=24)  # doctest: +SKIP
        10.0
    """
    if unit == "seconds":
        return value
    elif unit in ("frames", "samples"):
        return value / rate
    elif unit == "milliseconds":
        return value / 1000.0
    else:
        raise ValueError(f"Invalid time unit: {unit}")


def get_path_from_clipboard() -> str:
    """
    Get file path from clipboard and validate it.

    Intelligently extracts file paths from various clipboard contents including:
    - Direct file paths
    - File paths within error messages or tracebacks
    - Quoted paths
    """
    import os
    import re

    logger.info("Getting file path from clipboard...")
    clipboard_content = require_package("pyclip").paste()

    # Validate it's text, not binary data
    if isinstance(clipboard_content, bytes):
        try:
            file_path = clipboard_content.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(
                "Clipboard contains binary data that cannot be decoded as text. "
                "Expected a file path string."
            )
    elif isinstance(clipboard_content, str):
        file_path = clipboard_content
    else:
        raise ValueError(
            f"Clipboard content is not a valid string or bytes. "
            f"Got {type(clipboard_content).__name__}"
        )

    # Clean up the path
    file_path = os.path.expanduser(file_path.strip())

    # Try the path as-is first
    if os.path.isfile(file_path):
        logger.info("... File path: %s", file_path)
        return file_path

    # If that didn't work, try to extract paths from the text
    # Look for quoted paths or paths in error messages
    patterns = [
        r"'([^']+\.(?:mp4|mov|avi|mkv|m4v|webm|flv|wmv|mpg|mpeg))'",  # Single quoted paths
        r'"([^"]+\.(?:mp4|mov|avi|mkv|m4v|webm|flv|wmv|mpg|mpeg))"',  # Double quoted paths
        r"([/~][^\s'\"]+\.(?:mp4|mov|avi|mkv|m4v|webm|flv|wmv|mpg|mpeg))",  # Unquoted absolute paths
    ]

    for pattern in patterns:
        matches = re.findall(pattern, file_path, re.IGNORECASE)
        for match in matches:
            candidate = os.path.expanduser(match.strip())
            if os.path.isfile(candidate):
                logger.info(
                    "... Extracted file path from clipboard text: %s", candidate
                )
                return candidate

    # If still not found, give helpful error
    # Truncate long strings to avoid printing huge binary garbage
    display_content = file_path if len(file_path) < 100 else file_path[:100] + "..."
    raise ValueError(
        f"Clipboard content is not a valid (existing) file path: {display_content}\n"
        f"Tip: Copy the file path directly, not an error message containing it."
    )


def copy_to_clipboard(data: bytes | str) -> None:
    """
    Copy data to system clipboard.

    Args:
        data: Bytes (for binary data like images) or string to copy
    """
    pyclip = require_package("pyclip")
    pyclip.copy(data)
