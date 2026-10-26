"""ElevenLabs text-to-speech (TTS) HTTP client.

Stdlib-only — no ``requests`` or ``elevenlabs`` SDK dependency, mirroring
:mod:`mixing.transcript.scribe`. Synthesizes speech from text and lists the
voices available on the account.

The default model is ``eleven_multilingual_v2`` — ElevenLabs' high-quality,
multilingual model (handles English, French, and ~30 other languages with the
same voice), which is the right default for dubbing into multiple languages.

An optional on-disk cache (enabled by default here, since the same line is
often re-synthesized across runs) skips the network call when the same text +
voice + model + settings have been synthesized before.

Quick start:
    >>> from mixing.dubbing import list_voices, synthesize_to_file  # doctest: +SKIP
    >>> voices = list_voices()  # doctest: +SKIP
    >>> synthesize_to_file("Hello world", voices[0]["voice_id"], "hello.mp3")  # doctest: +SKIP
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Union

from mixing import _cache
from mixing._cache import CacheArg
from mixing._elevenlabs import resolve_api_key

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"
ELEVENLABS_SHARED_VOICES_URL = "https://api.elevenlabs.io/v1/shared-voices"
ELEVENLABS_ADD_VOICE_URL = "https://api.elevenlabs.io/v1/voices/add"
CACHE_ENV_KEY = "MIXING_TTS_CACHE_DIR"

#: High-quality, multilingual default model (one voice speaks many languages).
#:
#: **This is a DUBBING default, and deliberately not an expressive one.** A dub
#: has to track a performance that already exists: the timing is fixed by the
#: source, the delivery belongs to the original actor, and per-take variance is
#: a defect rather than life. ``eleven_multilingual_v2`` is steady and speaks
#: many languages in one voice, which is what that job wants.
#:
#: **If you are writing NARRATION or COMMENTARY, this is the wrong tool.** Use
#: :mod:`braidio`, which owns expressive delivery and defaults to ``eleven_v3``
#: so inline ``[audio tags]`` fire — the single biggest lever on whether a read
#: sounds alive (measured: plain text 70.8 Hz of pitch range, densely tagged
#: 124.4 Hz). v2 cannot render those tags at all, so a narration built on this
#: default is capped at the flat end before a word is written.
DFLT_MODEL_ID = "eleven_multilingual_v2"
#: Highest standard MP3 quality that does not require a paid tier.
DFLT_OUTPUT_FORMAT = "mp3_44100_128"

#: Voice settings for dubbing fidelity — see :data:`DFLT_MODEL_ID` on why this
#: is not tuned for expressiveness. ``stability`` ~0.5 keeps the read steady;
#: ``similarity_boost`` ~0.8 keeps timbre faithful to the chosen voice.
DFLT_VOICE_SETTINGS: dict[str, Any] = {
    "stability": 0.5,
    "similarity_boost": 0.8,
    "style": 0.0,
    "use_speaker_boost": True,
}

PathLike = Union[str, Path]


def default_cache_dir() -> Path:
    """Default on-disk cache for synthesized audio.

    Honors ``$MIXING_TTS_CACHE_DIR``, then ``$XDG_CACHE_HOME``, then
    ``~/.cache/``. Final segment is always ``mixing/tts``.
    """
    return _cache.default_cache_dir("tts", env_key=CACHE_ENV_KEY)


def text_to_speech(
    text: str,
    voice_id: str,
    *,
    api_key: str | None = None,
    model_id: str = DFLT_MODEL_ID,
    output_format: str = DFLT_OUTPUT_FORMAT,
    voice_settings: Mapping[str, Any] | None = None,
    language_code: str | None = None,
    timeout: float = 600.0,
    cache: CacheArg = True,
    refresh: bool = False,
    return_cache_status: bool = False,
) -> bytes | tuple[bytes, bool]:
    """Synthesize ``text`` to speech with ElevenLabs and return audio bytes.

    Args:
        text: The text to speak.
        voice_id: ElevenLabs voice id (see :func:`list_voices`).
        api_key: API key. Falls back to env var ``ELEVENLABS_API_KEY`` (or ``ELEVEN_API_KEY``).
        model_id: TTS model id. Defaults to ``eleven_multilingual_v2``.
        output_format: ElevenLabs ``output_format`` query value, e.g.
            ``mp3_44100_128`` (default), ``mp3_44100_192`` (needs a paid
            tier), or ``pcm_44100``.
        voice_settings: Override the default voice settings (stability,
            similarity_boost, style, use_speaker_boost). Merged over
            :data:`DFLT_VOICE_SETTINGS`.
        language_code: Optional ISO-639-1 hint (e.g. ``"fr"``). Only some
            models honor it; ignored by others.
        timeout: Total request timeout in seconds.
        cache: ``True`` (default) → use :func:`default_cache_dir`. ``False``
            → no cache. A path → use that directory. The cache key is a
            SHA-256 of the text plus every parameter that affects the audio.
        refresh: When ``True`` and ``cache`` is enabled, force a re-call and
            overwrite the cached entry.
        return_cache_status: When ``True``, return ``(audio, was_cached)`` instead
            of just ``audio``. ``was_cached`` is ``True`` iff the bytes came from the
            on-disk cache (no ElevenLabs call = no spend), so a caller can attribute
            real cost. Default ``False`` keeps the ``bytes`` return.

    Returns:
        Raw audio bytes in ``output_format`` (MP3 by default) — or, when
        ``return_cache_status`` is ``True``, a ``(audio, was_cached)`` tuple.

    Raises:
        RuntimeError: No API key supplied and ``ELEVENLABS_API_KEY`` (or ``ELEVEN_API_KEY``) unset.
        urllib.error.HTTPError: Request failed (4xx/5xx).
    """
    settings = {**DFLT_VOICE_SETTINGS, **(voice_settings or {})}

    cache_dir = _cache.resolve_cache_dir(cache, default_factory=default_cache_dir)
    if cache_dir is not None:
        key = _cache_key(
            text,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            settings=settings,
            language_code=language_code,
        )
        cached = _cache.read_cache(cache_dir, key, suffix=".audio")
        if cached is not None and not refresh:
            return (cached, True) if return_cache_status else cached

    api_key = resolve_api_key(api_key)

    payload: dict[str, Any] = {
        "text": text,
        "model_id": model_id,
        "voice_settings": settings,
    }
    if language_code:
        payload["language_code"] = language_code

    url = (
        f"{ELEVENLABS_TTS_URL}/{urllib.parse.quote(voice_id)}"
        f"?output_format={urllib.parse.quote(output_format)}"
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        audio = resp.read()

    if cache_dir is not None:
        _cache.write_cache(cache_dir, key, audio, suffix=".audio")
    return (audio, False) if return_cache_status else audio


def synthesize_to_file(
    text: str,
    voice_id: str,
    path: PathLike,
    **kwargs,
) -> Path:
    """Synthesize ``text`` and write the audio to ``path``.

    Thin convenience wrapper over :func:`text_to_speech`. Extra keyword
    arguments are forwarded unchanged.

    Returns:
        The path written.
    """
    audio = text_to_speech(text, voice_id, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio)
    return path


def list_voices(
    *,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """List the voices available on the account.

    Args:
        api_key: API key. Falls back to env var ``ELEVENLABS_API_KEY`` (or ``ELEVEN_API_KEY``).
        timeout: Request timeout in seconds.

    Returns:
        A list of voice dicts. Each has at least ``voice_id``, ``name``,
        ``category``, and a ``labels`` dict (``accent``, ``description``,
        ``age``, ``gender``, ``use_case``) describing the voice.

    Raises:
        RuntimeError: No API key supplied and ``ELEVENLABS_API_KEY`` (or ``ELEVEN_API_KEY``) unset.
    """
    api_key = resolve_api_key(api_key)
    req = urllib.request.Request(
        ELEVENLABS_VOICES_URL,
        method="GET",
        headers={"xi-api-key": api_key, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data.get("voices", [])


def search_shared_voices(
    *,
    language: str | None = None,
    use_cases: str | None = None,
    gender: str | None = None,
    category: str | None = None,
    sort: str | None = None,
    page_size: int = 40,
    api_key: str | None = None,
    timeout: float = 60.0,
    **extra_params,
) -> list[dict[str, Any]]:
    """Search ElevenLabs' public *shared voice library* (not just the account).

    Useful for finding a native voice in a target language (e.g. a French
    advertising voice) that is not yet on the account. Add the chosen voice
    with :func:`add_shared_voice`, then synthesize with the returned id.

    Args:
        language: ISO-639-1 language filter, e.g. ``"fr"``.
        use_cases: e.g. ``"advertisement"``, ``"narrative_story"``.
        gender: ``"male"`` / ``"female"`` / ``"neutral"``.
        category: e.g. ``"professional"``, ``"high_quality"``.
        sort: e.g. ``"trending"``.
        page_size: Max results.
        extra_params: Any other query params the endpoint accepts.

    Returns:
        A list of shared-voice dicts. Each has ``voice_id`` and
        ``public_owner_id`` (both needed by :func:`add_shared_voice`), plus
        ``name``, ``language``, ``accent``, ``gender``, ``use_case``,
        ``description``.
    """
    api_key = resolve_api_key(api_key)
    params = {"page_size": str(page_size)}
    for k, v in (
        ("language", language),
        ("use_cases", use_cases),
        ("gender", gender),
        ("category", category),
        ("sort", sort),
    ):
        if v is not None:
            params[k] = v
    params.update({k: str(v) for k, v in extra_params.items()})
    url = f"{ELEVENLABS_SHARED_VOICES_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url, method="GET", headers={"xi-api-key": api_key, "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data.get("voices", [])


def add_shared_voice(
    public_owner_id: str,
    voice_id: str,
    name: str,
    *,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> str:
    """Add a shared-library voice to the account so it can be synthesized.

    Idempotent-ish: if the voice (by name) is already on the account, returns
    its existing id instead of erroring on a duplicate add.

    Args:
        public_owner_id: ``public_owner_id`` from :func:`search_shared_voices`.
        voice_id: The shared voice's ``voice_id``.
        name: Name to give the added voice on the account.
        api_key: API key. Falls back to env var ``ELEVENLABS_API_KEY`` (or ``ELEVEN_API_KEY``).

    Returns:
        The account-local ``voice_id`` to pass to :func:`text_to_speech`.
    """
    api_key = resolve_api_key(api_key)

    existing = find_voice(name, api_key=api_key)
    if existing is not None:
        return existing["voice_id"]

    url = f"{ELEVENLABS_ADD_VOICE_URL}/{urllib.parse.quote(public_owner_id)}/{urllib.parse.quote(voice_id)}"
    req = urllib.request.Request(
        url,
        data=json.dumps({"new_name": name}).encode(),
        method="POST",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data["voice_id"]


def find_voice(
    query: str,
    *,
    api_key: str | None = None,
    voices: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return the first voice whose name or labels contain ``query`` (case-insensitive).

    Convenience for picking a voice by a human name (e.g. ``"Brian"``) or a
    label keyword (e.g. ``"french"``) without memorizing voice ids.
    """
    q = query.lower()
    voices = voices if voices is not None else list_voices(api_key=api_key)
    for v in voices:
        hay = " ".join(
            [str(v.get("name", "")), json.dumps(v.get("labels", {}))]
        ).lower()
        if q in hay:
            return v
    return None


def _cache_key(
    text: str,
    *,
    voice_id: str,
    model_id: str,
    output_format: str,
    settings: Mapping[str, Any],
    language_code: str | None,
) -> str:
    """Cache key over the text and every parameter that affects the audio."""
    return _cache.sha256_key(
        text,
        voice_id,
        model_id,
        output_format,
        language_code or "",
        json.dumps(settings, sort_keys=True),
    )
