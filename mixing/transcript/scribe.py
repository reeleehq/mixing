"""ElevenLabs Scribe (speech-to-text) HTTP client.

Stdlib-only — no ``requests`` or ``elevenlabs`` SDK dependency, so this
module adds nothing to the project's required deps. Returns the raw
JSON response which includes word-level timestamps (with per-word
``confidence``) when ``timestamps_granularity="word"`` (the default).

Optional on-disk cache: pass ``cache=True`` (default location) or
``cache=<path>`` to skip a re-call when the same audio + params have
been transcribed before.
"""

from __future__ import annotations

import json
import mimetypes
import os
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Union

from mixing import _cache
from mixing._cache import CacheArg
from mixing._elevenlabs import resolve_api_key

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
CACHE_ENV_KEY = "MIXING_TRANSCRIPT_CACHE_DIR"

PathLike = Union[str, Path]
AudioInput = Union[PathLike, bytes]


def default_cache_dir() -> Path:
    """Default on-disk cache for Scribe responses.

    Honors ``$MIXING_TRANSCRIPT_CACHE_DIR``, then ``$XDG_CACHE_HOME``,
    then ``~/.cache/``. Final segment is always ``mixing/transcript``.
    """
    return _cache.default_cache_dir("transcript", env_key=CACHE_ENV_KEY)


def transcribe(
    audio: AudioInput,
    *,
    api_key: str | None = None,
    model_id: str = "scribe_v1",
    timestamps_granularity: str = "word",
    tag_audio_events: bool = True,
    diarize: bool = False,
    language_code: str | None = None,
    extra_fields: Mapping[str, str] | None = None,
    timeout: float = 600.0,
    cache: CacheArg = False,
    refresh: bool = False,
) -> dict[str, Any]:
    """Transcribe ``audio`` with ElevenLabs Scribe.

    Args:
        audio: Path to an audio/video file, or raw bytes.
        api_key: API key. Falls back to env var ``ELEVENLABS_API_KEY`` (or ``ELEVEN_API_KEY``).
        model_id: Scribe model id (currently ``scribe_v1``).
        timestamps_granularity: ``"word"`` (default), ``"character"``, or ``"none"``.
        tag_audio_events: Whether to surface ``(laughs)`` / ``(coughs)`` etc.
        diarize: Speaker diarization on/off.
        language_code: Optional BCP-47 hint to skip language detection.
        extra_fields: Additional multipart fields to send (forward compatible).
        timeout: Total request timeout in seconds.
        cache: ``False`` (default) → no cache. ``True`` → use
            :func:`default_cache_dir`. A path → use that directory.
            The cache key is the SHA-256 of the audio bytes plus all
            request parameters that affect the response.
        refresh: When ``True`` and ``cache`` is enabled, force a re-call
            and overwrite the cached entry. Useful for invalidating
            stale entries when Scribe upgrades its model.

    Returns:
        The raw JSON response from ElevenLabs (a dict). When
        ``timestamps_granularity="word"`` (default), the response contains
        a ``"words"`` list where each entry has at minimum
        ``{"text", "start", "end", "type", "confidence"}``. Non-word
        events (``type != "word"``) include ``(laughs)`` etc. when
        ``tag_audio_events=True``.

    Raises:
        RuntimeError: No API key supplied and ``ELEVENLABS_API_KEY`` (or ``ELEVEN_API_KEY``) unset.
        urllib.error.HTTPError: Request failed (4xx/5xx).
    """
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        audio_bytes = path.read_bytes()
        filename = path.name
    else:
        audio_bytes = audio
        filename = "audio.bin"

    cache_dir = _cache.resolve_cache_dir(cache, default_factory=default_cache_dir)

    if cache_dir is not None:
        key = _cache_key(
            audio_bytes,
            model_id=model_id,
            timestamps_granularity=timestamps_granularity,
            tag_audio_events=tag_audio_events,
            diarize=diarize,
            language_code=language_code,
            extra_fields=extra_fields,
        )
        cached = _cache.read_cache(cache_dir, key, suffix=".json", loads=_json_loads)
        if cached is not None and not refresh:
            return cached

    api_key = resolve_api_key(api_key)

    fields: dict[str, str] = {
        "model_id": model_id,
        "timestamps_granularity": timestamps_granularity,
        "tag_audio_events": str(tag_audio_events).lower(),
        "diarize": str(diarize).lower(),
    }
    if language_code:
        fields["language_code"] = language_code
    if extra_fields:
        fields.update({k: str(v) for k, v in extra_fields.items()})

    body, content_type = _multipart_encode(fields, filename, audio_bytes)
    req = urllib.request.Request(
        ELEVENLABS_STT_URL,
        data=body,
        method="POST",
        headers={"xi-api-key": api_key, "Content-Type": content_type},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        response = json.loads(resp.read().decode())

    if cache_dir is not None:
        _cache.write_cache(cache_dir, key, response, suffix=".json", dumps=_json_dumps)
    return response


def _json_loads(data: bytes) -> dict[str, Any]:
    return json.loads(data.decode())


def _json_dumps(value: object) -> bytes:
    return json.dumps(value).encode()


def _cache_key(
    audio_bytes: bytes,
    *,
    model_id: str,
    timestamps_granularity: str,
    tag_audio_events: bool,
    diarize: bool,
    language_code: str | None,
    extra_fields: Mapping[str, str] | None,
) -> str:
    extra = ""
    if extra_fields:
        extra = "".join(f"\0{k}={extra_fields[k]}" for k in sorted(extra_fields))
    return _cache.sha256_key(
        audio_bytes,
        model_id,
        timestamps_granularity,
        "1" if tag_audio_events else "0",
        # NOTE: kept adjacent (no delimiter) to match the original key scheme.
        ("1" if diarize else "0") + (language_code or "") + extra,
    )


def _multipart_encode(
    fields: Mapping[str, str], filename: str, file_bytes: bytes
) -> tuple[bytes, str]:
    boundary = "----mixingTranscript" + os.urandom(8).hex()
    head_parts: list[bytes] = []
    for k, v in fields.items():
        head_parts.append(f"--{boundary}\r\n".encode())
        head_parts.append(
            f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        )
    file_mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    head_parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {file_mime}\r\n\r\n"
        ).encode()
    )
    closing = f"\r\n--{boundary}--\r\n".encode()
    body = b"".join(head_parts) + file_bytes + closing
    return body, f"multipart/form-data; boundary={boundary}"
