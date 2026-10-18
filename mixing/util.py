"""General utilities for mixing video, audio, etc."""

from typing import Literal
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
MEDIA_INSTALL_HINT = f"pip install 'mixing[{MEDIA_EXTRA}]'"

# Import names (top-level) that the ``media`` extra provides, per media area.
AUDIO_IMPORT_NAMES = ("numpy", "pydub", "scipy")
VIDEO_IMPORT_NAMES = ("numpy", "moviepy", "cv2", "PIL", "burns", "looks")
# Everything the extra supplies; ``require_package`` uses it to pick its hint.
MEDIA_IMPORT_NAMES = tuple(dict.fromkeys(AUDIO_IMPORT_NAMES + VIDEO_IMPORT_NAMES))

# The ``ml`` extra: pretrained models (the AudioSet tagger behind the "audioset"
# segmentation strategy). Imported lazily, inside the functions that need them.
ML_EXTRA = "ml"
ML_INSTALL_HINT = f"pip install 'mixing[{ML_EXTRA}]'"
ML_IMPORT_NAMES = ("torch", "transformers")


def _is_importable(name: str) -> bool:
    """True if top-level module ``name`` can be found, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def require_media_extra(feature: str, *, modules: tuple[str, ...] = MEDIA_IMPORT_NAMES):
    """Raise an informative ``ImportError`` if the ``media`` extra is not installed.

    The base ``pip install mixing`` ships only the pure modules (``srt``,
    ``chapters``, ``errors``, ``util``, ``egress``, ``transcript``, TTS); the
    moviepy / opencv / scipy / pydub backends come with ``mixing[media]``.

    >>> require_media_extra("demo", modules=("math",))
    >>> require_media_extra("demo", modules=("no_such_module_xyz",))
    Traceback (most recent call last):
      ...
    ImportError: demo requires the 'media' extra ...
    """
    _require_extra(feature, extra=MEDIA_EXTRA, hint=MEDIA_INSTALL_HINT, modules=modules)


def require_ml_extra(feature: str, *, modules: tuple[str, ...] = ML_IMPORT_NAMES):
    """Raise an informative ``ImportError`` if the ``ml`` extra is not installed.

    ``torch`` / ``transformers`` (the pretrained AudioSet tagger) come with
    ``mixing[ml]``; nothing imports them until a learned feature is used.

    >>> require_ml_extra("demo", modules=("math",))
    >>> require_ml_extra("demo", modules=("no_such_module_xyz",))
    Traceback (most recent call last):
      ...
    ImportError: demo requires the 'ml' extra ...
    """
    _require_extra(feature, extra=ML_EXTRA, hint=ML_INSTALL_HINT, modules=modules)


def _require_extra(feature: str, *, extra: str, hint: str, modules: tuple[str, ...]):
    missing = [name for name in modules if not _is_importable(name)]
    if missing:
        raise ImportError(
            f"{feature} requires the {extra!r} extra "
            f"(missing: {', '.join(missing)}). Install it with "
            f"{hint}."
        )


def require_package(package_name: str):
    """
    Import a package, raising an informative error if not installed.

    Packages supplied by the ``media`` or ``ml`` extra get a hint naming that extra.

    >>> math = require_package('math')  # doctest: +SKIP
    >>> math.pi  # doctest: +SKIP
    3.141592653589793
    """
    try:
        return importlib.import_module(package_name)
    except ImportError as e:
        top = package_name.split(".")[0]
        if top in MEDIA_IMPORT_NAMES:
            hint = MEDIA_INSTALL_HINT
        elif top in ML_IMPORT_NAMES:
            hint = ML_INSTALL_HINT
        else:
            hint = f"pip install {package_name}"
        raise ImportError(
            f"Package '{package_name}' is required for this functionality. "
            f"Please install it via '{hint}'."
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
