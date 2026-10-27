"""A jupyter notebook widget for simple audio playback, visualization, and editing."""

import io
import base64
import logging
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional

from ..util import WIDGET_EXTRA, install_hint

try:
    from IPython.display import display, HTML
except ImportError as e:
    raise ImportError(
        f"The audio widget needs Jupyter: {install_hint(WIDGET_EXTRA)}", name=e.name
    ) from e

logger = logging.getLogger(__name__)


def _soundfile():
    """``soundfile``, or an ``ImportError`` naming the ``widget`` extra."""
    try:
        import soundfile
    except ImportError as e:
        raise ImportError(
            f"soundfile is required for the audio widget: {install_hint(WIDGET_EXTRA)}",
            name="soundfile",
        ) from e
    return soundfile


# Type aliases
WaveformArray = (
    np.ndarray
)  # Shape: (n_samples,) for mono or (n_samples, n_channels) for stereo
SampleRate = int
WfSr = Tuple[WaveformArray, SampleRate]
AudioInput = Union[str, Path, bytes, WfSr]


def ensure_wfsr(audio_input: AudioInput) -> WfSr:
    """
    Convert various audio input types to standardized (wf, sr) tuple.

    This function accepts multiple input formats and normalizes them to a standard
    NumPy array waveform and integer sample rate tuple.

    Args:
        audio_input: One of:
            - str or Path: File path to audio file
            - bytes: Raw audio file bytes
            - tuple: Already in (wf, sr) format

    Returns:
        Tuple of (waveform, sample_rate) where:
            - waveform is a NumPy array with shape (n_samples,) or (n_samples, n_channels)
            - sample_rate is an integer representing Hz

    Raises:
        ImportError: If soundfile is not installed
        TypeError: If input type is not supported
        FileNotFoundError: If file path doesn't exist

    Examples:
        >>> wf, sr = ensure_wfsr("audio.wav")  # doctest: +SKIP
        >>> wf, sr = ensure_wfsr(Path("audio.mp3"))  # doctest: +SKIP
        >>> wf, sr = ensure_wfsr(audio_bytes)  # doctest: +SKIP
        >>> wf, sr = ensure_wfsr((wf, sr))  # doctest: +SKIP
    """
    sf = _soundfile()

    # Already in (wf, sr) format
    if isinstance(audio_input, tuple) and len(audio_input) == 2:
        wf, sr = audio_input
        if isinstance(wf, np.ndarray) and isinstance(sr, (int, np.integer)):
            return wf, int(sr)
        else:
            raise TypeError(
                f"Invalid (wf, sr) tuple: expected (ndarray, int), "
                f"got ({type(wf).__name__}, {type(sr).__name__})"
            )

    # File path (str or Path)
    elif isinstance(audio_input, (str, Path)):
        path = Path(audio_input)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        wf, sr = sf.read(str(path))
        return wf, int(sr)

    # Raw bytes
    elif isinstance(audio_input, bytes):
        wf, sr = sf.read(io.BytesIO(audio_input))
        return wf, int(sr)

    else:
        raise TypeError(
            f"Unsupported audio input type: {type(audio_input).__name__}. "
            f"Expected str, Path, bytes, or (wf, sr) tuple."
        )


def wf_to_base64_wav(wf: WaveformArray, sr: SampleRate) -> str:
    """
    Convert waveform array to base64-encoded WAV data URL.

    Args:
        wf: Waveform array
        sr: Sample rate in Hz

    Returns:
        Base64-encoded data URL string suitable for HTML audio element
    """
    sf = _soundfile()

    # Ensure wf is in proper format
    if wf.dtype != np.float32 and wf.dtype != np.float64:
        wf = wf.astype(np.float32)

    # Write to bytes buffer
    buffer = io.BytesIO()
    sf.write(buffer, wf, sr, format="WAV")
    buffer.seek(0)

    # Encode to base64
    audio_bytes = buffer.read()
    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

    return f"data:audio/wav;base64,{b64_audio}"


def wf_to_json_array(wf: WaveformArray, max_points: int = 2000) -> str:
    """
    Convert waveform to JSON array for visualization, downsampled if needed.

    Args:
        wf: Waveform array
        max_points: Maximum number of points to include

    Returns:
        JSON array string
    """
    # Convert to mono if stereo (take mean)
    if wf.ndim == 2:
        wf = wf.mean(axis=1)

    # Downsample if needed
    if len(wf) > max_points:
        step = len(wf) // max_points
        wf = wf[::step]

    # Normalize to [-1, 1]
    if wf.max() > 0:
        wf = wf / max(abs(wf.min()), abs(wf.max()))

    import json

    return json.dumps(wf.tolist())


class AudioWidget:
    """
    Interactive audio widget for Jupyter notebooks with DAW-like features.

    This widget provides:
    - Interactive waveform visualization
    - Playback controls (play/pause with synchronized playhead)
    - Time-based segment selection
    - Zoom operation (focus on selected segment)
    - Crop operation (trim to selected segment)
    - Extensible architecture for custom visualizations and operations

    Args:
        audio_input: Audio data in any format supported by ensure_wfsr
        height: Height of waveform display in pixels
        waveform_color: Color of waveform bars
        progress_color: Color of playhead/progress indicator

    Examples:
        >>> widget = AudioWidget("audio.wav")  # doctest: +SKIP
        >>> widget  # Will auto-display in notebook  # doctest: +SKIP
        >>> # User selects region in UI, then:
        >>> widget.set_selection(2.0, 5.0)  # Set selection programmatically
        >>> widget.crop()  # Crop to selected region  # doctest: +SKIP
    """

    def __init__(
        self,
        audio_input: AudioInput,
        height: float = 128,
        waveform_color: str = "#4A90E2",
        progress_color: str = "#FF6B6B",
    ):
        """Initialize the audio widget with audio data."""
        # Convert input to standard format
        self.wf, self.sr = ensure_wfsr(audio_input)

        # Store original for reset
        self._original_wf = self.wf.copy()
        self._original_sr = self.sr

        # Set display properties
        self.height = height
        self.waveform_color = waveform_color
        self.progress_color = progress_color

        # Selection state (in seconds)
        self.selection_start = 0.0
        self.selection_end = 0.0
        self.has_selection = False

        # Generate unique ID for this widget instance
        import random

        self._id = f"aw_{random.randint(10000, 99999)}"

        # Create the HTML widget
        self._html_widget = None
        self._create_widget()

    @property
    def duration(self) -> float:
        """Duration of current audio in seconds."""
        return len(self.wf) / self.sr

    def _create_widget(self):
        """Create the HTML widget with canvas-based waveform."""
        audio_data_url = wf_to_base64_wav(self.wf, self.sr)
        waveform_data = wf_to_json_array(self.wf)

        html_content = f"""
        <div id="{self._id}" style="width: 100%; padding: 15px; background-color: #f8f9fa; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">

            <!-- Waveform Canvas -->
            <canvas id="{self._id}_canvas" width="800" height="{int(self.height)}"
                    style="width: 100%; height: {int(self.height)}px; background-color: #ffffff; border-radius: 4px; cursor: crosshair; display: block; margin-bottom: 10px;">
            </canvas>

            <!-- Audio Element (hidden) -->
            <audio id="{self._id}_audio" preload="auto" style="display: none;">
                <source src="{audio_data_url}" type="audio/wav">
            </audio>

            <!-- Controls -->
            <div style="display: flex; gap: 10px; align-items: center; margin-top: 10px;">
                <button id="{self._id}_play" style="padding: 8px 16px; border: none; border-radius: 4px; background-color: #4A90E2; color: white; cursor: pointer; font-weight: 600; font-size: 14px;">
                    ▶ Play
                </button>
                <button id="{self._id}_stop" style="padding: 8px 16px; border: none; border-radius: 4px; background-color: #6c757d; color: white; cursor: pointer; font-weight: 600; font-size: 14px;">
                    ■ Stop
                </button>
                <span id="{self._id}_time" style="font-family: 'Monaco', 'Courier New', monospace; font-size: 14px; color: #495057;">
                    0.00s / {self.duration:.2f}s
                </span>
                <span id="{self._id}_selection" style="font-family: 'Monaco', 'Courier New', monospace; font-size: 12px; color: #6c757d; margin-left: auto;">
                    No selection
                </span>
            </div>

            <!-- Info Bar -->
            <div style="margin-top: 10px; padding: 8px 12px; background-color: #e9ecef; border-radius: 4px; font-size: 12px; color: #495057;">
                <strong>Duration:</strong> {self.duration:.2f}s |
                <strong>Sample Rate:</strong> {self.sr} Hz |
                <strong>Channels:</strong> {1 if self.wf.ndim == 1 else self.wf.shape[1]} |
                <strong>Samples:</strong> {len(self.wf):,}
            </div>
        </div>

        <script>
        (function() {{
            const canvas = document.getElementById('{self._id}_canvas');
            const ctx = canvas.getContext('2d');
            const audio = document.getElementById('{self._id}_audio');
            const playButton = document.getElementById('{self._id}_play');
            const stopButton = document.getElementById('{self._id}_stop');
            const timeDisplay = document.getElementById('{self._id}_time');
            const selectionDisplay = document.getElementById('{self._id}_selection');

            const waveformData = {waveform_data};
            const duration = {self.duration};
            const waveColor = '{self.waveform_color}';
            const progressColor = '{self.progress_color}';

            let selectionStart = null;
            let selectionEnd = null;
            let isSelecting = false;

            // Set canvas resolution
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);

            // Draw waveform
            function drawWaveform() {{
                const width = rect.width;
                const height = rect.height;
                const mid = height / 2;

                // Clear canvas
                ctx.clearRect(0, 0, width, height);

                // Draw waveform
                ctx.strokeStyle = waveColor;
                ctx.lineWidth = 1;
                ctx.beginPath();

                for (let i = 0; i < waveformData.length; i++) {{
                    const x = (i / waveformData.length) * width;
                    const y = mid + (waveformData[i] * mid * 0.9);
                    if (i === 0) {{
                        ctx.moveTo(x, y);
                    }} else {{
                        ctx.lineTo(x, y);
                    }}
                }}
                ctx.stroke();

                // Draw center line
                ctx.strokeStyle = '#dee2e6';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(0, mid);
                ctx.lineTo(width, mid);
                ctx.stroke();

                // Draw selection if exists
                if (selectionStart !== null && selectionEnd !== null) {{
                    const startX = (selectionStart / duration) * width;
                    const endX = (selectionEnd / duration) * width;

                    ctx.fillStyle = 'rgba(255, 107, 107, 0.2)';
                    ctx.fillRect(startX, 0, endX - startX, height);

                    ctx.strokeStyle = '#FF6B6B';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(startX, 0);
                    ctx.lineTo(startX, height);
                    ctx.moveTo(endX, 0);
                    ctx.lineTo(endX, height);
                    ctx.stroke();
                }}

                // Draw playhead
                if (!audio.paused) {{
                    const playheadX = (audio.currentTime / duration) * width;
                    ctx.strokeStyle = progressColor;
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(playheadX, 0);
                    ctx.lineTo(playheadX, height);
                    ctx.stroke();
                }}
            }}

            drawWaveform();

            // Play/Pause
            playButton.onclick = function() {{
                if (audio.paused) {{
                    audio.play();
                    playButton.innerHTML = '⏸ Pause';
                }} else {{
                    audio.pause();
                    playButton.innerHTML = '▶ Play';
                }}
            }};

            // Stop
            stopButton.onclick = function() {{
                audio.pause();
                audio.currentTime = 0;
                playButton.innerHTML = '▶ Play';
                drawWaveform();
            }};

            // Time update
            audio.ontimeupdate = function() {{
                const current = audio.currentTime;
                timeDisplay.innerHTML = current.toFixed(2) + 's / ' + duration.toFixed(2) + 's';
                drawWaveform();
            }};

            audio.onended = function() {{
                playButton.innerHTML = '▶ Play';
                drawWaveform();
            }};

            // Selection handling
            canvas.onmousedown = function(e) {{
                const x = e.offsetX;
                const clickTime = (x / rect.width) * duration;
                selectionStart = clickTime;
                selectionEnd = clickTime;
                isSelecting = true;
                drawWaveform();
            }};

            canvas.onmousemove = function(e) {{
                if (isSelecting) {{
                    const x = e.offsetX;
                    selectionEnd = (x / rect.width) * duration;
                    drawWaveform();

                    const start = Math.min(selectionStart, selectionEnd);
                    const end = Math.max(selectionStart, selectionEnd);
                    selectionDisplay.innerHTML =
                        'Selection: ' + start.toFixed(2) + 's - ' + end.toFixed(2) + 's ' +
                        '(' + (end - start).toFixed(2) + 's)';
                }}
            }};

            canvas.onmouseup = function() {{
                if (isSelecting) {{
                    isSelecting = false;

                    // Ensure start < end
                    if (selectionStart > selectionEnd) {{
                        [selectionStart, selectionEnd] = [selectionEnd, selectionStart];
                    }}

                    // Store in window for Python access
                    window.{self._id}_selection = {{
                        start: selectionStart,
                        end: selectionEnd
                    }};

                    console.log('Selection:', selectionStart, '-', selectionEnd);
                }}
            }};

            canvas.onmouseleave = function() {{
                if (isSelecting) {{
                    isSelecting = false;
                }}
            }};

            // Make functions available globally
            window.{self._id}_drawWaveform = drawWaveform;
        }})();
        </script>
        """

        self._html_widget = HTML(html_content)

    def display(self):
        """Display the widget in the notebook."""
        display(self._html_widget)
        return self

    def _ipython_display_(self):
        """Support for automatic display when returned in a cell."""
        self.display()

    def _get_js_selection(self):
        """Try to get selection from JavaScript."""
        from IPython.display import Javascript

        js_code = f"""
        (function() {{
            const sel = window.{self._id}_selection;
            if (sel) {{
                const code = `
__temp_sel_start = ${{sel.start}}
__temp_sel_end = ${{sel.end}}
`;
                IPython.notebook.kernel.execute(code);
            }}
        }})();
        """
        display(Javascript(js_code))

        # Wait briefly for execution
        import time

        time.sleep(0.15)

        # Try to retrieve from globals
        try:
            if "__temp_sel_start" in globals() and "__temp_sel_end" in globals():
                self.selection_start = globals()["__temp_sel_start"]
                self.selection_end = globals()["__temp_sel_end"]
                self.has_selection = self.selection_start < self.selection_end
                # Clean up
                del globals()["__temp_sel_start"]
                del globals()["__temp_sel_end"]
                return True
        except:
            pass

        return False

    def zoom(self):
        """
        Zoom the display to show only the selected time segment.

        Raises:
            ValueError: If no valid selection exists
        """
        # Try to get selection from JS
        self._get_js_selection()

        if not self.has_selection:
            raise ValueError(
                "No selection made. Please select a region by clicking and dragging on the waveform, "
                "or use widget.set_selection(start, end) first."
            )

        if self.selection_start >= self.selection_end:
            raise ValueError(
                f"Invalid selection: start ({self.selection_start:.2f}s) "
                f"must be before end ({self.selection_end:.2f}s)"
            )

        # Calculate sample indices
        start_sample = int(self.selection_start * self.sr)
        end_sample = int(self.selection_end * self.sr)

        # Ensure within bounds
        start_sample = max(0, min(start_sample, len(self.wf)))
        end_sample = max(start_sample, min(end_sample, len(self.wf)))

        # Extract segment
        self.wf = self.wf[start_sample:end_sample].copy()

        # Reset selection
        self.selection_start = 0.0
        self.selection_end = 0.0
        self.has_selection = False

        # Recreate and display widget
        self._create_widget()
        self.display()

        logger.info("Zoomed to %.2fs segment", (end_sample - start_sample) / self.sr)

    def crop(self):
        """
        Crop the audio data to the selected time segment.

        Raises:
            ValueError: If no valid selection exists
        """
        # Try to get selection from JS
        self._get_js_selection()

        if not self.has_selection:
            raise ValueError(
                "No selection made. Please select a region by clicking and dragging on the waveform, "
                "or use widget.set_selection(start, end) first."
            )

        if self.selection_start >= self.selection_end:
            raise ValueError(
                f"Invalid selection: start ({self.selection_start:.2f}s) "
                f"must be before end ({self.selection_end:.2f}s)"
            )

        # Calculate sample indices
        start_sample = int(self.selection_start * self.sr)
        end_sample = int(self.selection_end * self.sr)

        # Ensure within bounds
        start_sample = max(0, min(start_sample, len(self.wf)))
        end_sample = max(start_sample, min(end_sample, len(self.wf)))

        # Crop the waveform
        self.wf = self.wf[start_sample:end_sample].copy()

        # Reset selection
        self.selection_start = 0.0
        self.selection_end = 0.0
        self.has_selection = False

        # Recreate and display widget
        self._create_widget()
        self.display()

        logger.info(
            "Cropped to %.2fs segment (%s samples)",
            (end_sample - start_sample) / self.sr,
            f"{end_sample - start_sample:,}",
        )

    def set_selection(self, start: float, end: float):
        """
        Set the selection programmatically.

        Args:
            start: Selection start time in seconds
            end: Selection end time in seconds

        Returns:
            Self for method chaining
        """
        if start >= end:
            raise ValueError(
                f"Start time ({start}s) must be less than end time ({end}s)"
            )

        if start < 0 or end > self.duration:
            raise ValueError(
                f"Selection must be within audio duration (0s - {self.duration:.2f}s)"
            )

        self.selection_start = start
        self.selection_end = end
        self.has_selection = True
        logger.info("Selection set: %.2fs - %.2fs (%.2fs)", start, end, end - start)
        return self

    def reset(self):
        """Reset the audio to its original state."""
        self.wf = self._original_wf.copy()
        self.sr = self._original_sr

        self.selection_start = 0.0
        self.selection_end = 0.0
        self.has_selection = False

        # Recreate and display widget
        self._create_widget()
        self.display()

        logger.info("Audio reset to original state")

    def get_waveform(self) -> WfSr:
        """
        Get the current waveform data.

        Returns:
            Tuple of (waveform, sample_rate)
        """
        return self.wf, self.sr

    def get_selection(self) -> Optional[WfSr]:
        """
        Get the currently selected audio segment as (wf, sr) tuple.

        Returns:
            Tuple of (waveform, sample_rate) for selected region, or None if no selection
        """
        if not self.has_selection:
            return None

        start_sample = int(self.selection_start * self.sr)
        end_sample = int(self.selection_end * self.sr)

        # Ensure within bounds
        start_sample = max(0, min(start_sample, len(self.wf)))
        end_sample = max(start_sample, min(end_sample, len(self.wf)))

        return self.wf[start_sample:end_sample].copy(), self.sr

    def save(self, filepath: Union[str, Path], format: Optional[str] = None):
        """
        Save the current audio to a file.

        Args:
            filepath: Output file path
            format: Audio format (auto-detected from extension if None)
        """
        sf = _soundfile()

        filepath = Path(filepath)

        # Auto-detect format
        if format is None:
            format = filepath.suffix[1:] if filepath.suffix else "WAV"

        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save
        sf.write(str(filepath), self.wf, self.sr, format=format.upper())
        logger.info("Saved audio to: %s", filepath)

    def apply_fade_in(self, duration: float = 1.0):
        """
        Apply fade-in effect to the current audio.

        Args:
            duration: Fade duration in seconds
        """
        fade_samples = int(duration * self.sr)
        fade_samples = min(fade_samples, len(self.wf))

        # Create fade curve
        fade_curve = np.linspace(0, 1, fade_samples).astype(self.wf.dtype)

        # Apply fade
        if self.wf.ndim == 1:
            self.wf[:fade_samples] *= fade_curve
        else:
            self.wf[:fade_samples] *= fade_curve[:, np.newaxis]

        # Recreate and display widget
        self._create_widget()
        self.display()

        logger.info("Applied %ss fade-in", duration)

    def apply_fade_out(self, duration: float = 1.0):
        """
        Apply fade-out effect to the current audio.

        Args:
            duration: Fade duration in seconds
        """
        fade_samples = int(duration * self.sr)
        fade_samples = min(fade_samples, len(self.wf))

        # Create fade curve
        fade_curve = np.linspace(1, 0, fade_samples).astype(self.wf.dtype)

        # Apply fade
        if self.wf.ndim == 1:
            self.wf[-fade_samples:] *= fade_curve
        else:
            self.wf[-fade_samples:] *= fade_curve[:, np.newaxis]

        # Recreate and display widget
        self._create_widget()
        self.display()

        logger.info("Applied %ss fade-out", duration)
