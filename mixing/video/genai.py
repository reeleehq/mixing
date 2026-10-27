"""
AI Video Generation with Google Vertex AI Veo

This module provides tools for generating videos using Google's Veo models on Vertex AI.
The main workflow is simple: **generate and get a file path** to your video.

It is an *optional* feature: it requires ``google-genai`` (``pip install
'mixing[gen]'``) and Google Cloud credentials. Importing it pulls
``from google import genai``, so it is not wired into the lazy ``mixing.video``
facade — import it explicitly via ``mixing.video.genai``.

Key features include:

• **One-Step Generation**: Generate videos and get file paths in one call
• **Flexible Egress**: Control how videos are saved with the `output` parameter
• **Multiple Input Formats**: Generate from text prompts, images, or video frames
• **Smart Authentication**: Automatic environment variable detection and helpful error messages
• **Multiple Video Support**: Handle models that generate multiple video variations

Quick Start (Generate and Save):
```python
from mixing.video.genai import generate_video

# Generate video and get file path in one call
video_path = generate_video("A serene forest at dawn with mist")
print(f"Video saved to: {video_path}")
```

Flexible Egress Options:
```python
# Default: Auto-save to temp files and return path(s)
path = generate_video("prompt")

# Save to specific location
path = generate_video("prompt", output="/path/to/my_video.mp4")

# Save to directory with auto-naming
path = generate_video("prompt", output="/path/to/output_dir/")

# Just return the operation (no saving)
op = generate_video("prompt", output=False)
# or: op = generate_video("prompt", output=lambda x: x)

# Custom processing function
def my_processor(operation):
    paths = save_generated_videos(operation, prefix="custom_")
    print(f"Saved {len(paths)} videos")
    return paths

result = generate_video("prompt", output=my_processor)
```

Authentication Setup:
Set environment variables for automatic authentication:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

Main Functions:
- generate_video(): Generate videos with flexible egress options
- save_generated_videos(): Standalone video saving utility
- Frame extraction utilities for using video frames as inputs
"""

import time
import base64
import logging
import mimetypes
import os
import tempfile
from typing import Union
from collections.abc import Callable

from dol import non_colliding_key

from ..util import require_package
from ._helpers import _is_video_file

logger = logging.getLogger(__name__)


def _log_auth_help():
    """Log helpful authentication setup information."""
    help_msg = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        GOOGLE CLOUD AUTHENTICATION SETUP                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║ To authenticate with Google Cloud Vertex AI, you have several options:      ║
║                                                                              ║
║ 1. SERVICE ACCOUNT (Recommended for production):                            ║
║    • Download service account JSON from Google Cloud Console                ║
║    • Set environment variable:                                              ║
║      export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"  ║
║    • Or set: export VEO_SERVICE_ACCOUNT_FILE="/path/to/service-account.json"║
║                                                                              ║
║ 2. PROJECT ID (Required):                                                   ║
║    • Set environment variable:                                              ║
║      export GOOGLE_CLOUD_PROJECT="your-project-id"                          ║
║    • Or set: export VEO_PROJECT_ID="your-project-id"                        ║
║                                                                              ║
║ 3. APPLICATION DEFAULT CREDENTIALS (Development):                           ║
║    • Run: gcloud auth application-default login                             ║
║    • This allows local development without service account                  ║
║                                                                              ║
║ 4. FUNCTION PARAMETERS (Override):                                          ║
║    • Pass service_account_file="path/to/file.json"                          ║
║    • Pass project_id="your-project-id"                                      ║
║                                                                              ║
║ For more info: https://cloud.google.com/docs/authentication                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    logger.error(help_msg)


def _get_auth_from_env():
    """Get authentication info from environment variables."""
    # Check for service account file
    service_account_file = os.getenv("VEO_SERVICE_ACCOUNT_FILE") or os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )

    # Check for project ID
    project_id = (
        os.getenv("VEO_PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
    )

    # Check for location
    location = (
        os.getenv("VEO_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"
    )

    return service_account_file, project_id, location


def _setup_genai_client(
    service_account_file=None, project_id=None, location="us-central1"
):
    """Set up Google GenAI client with proper error handling."""
    genai = require_package("google.genai")

    try:
        if service_account_file:
            from google.oauth2 import service_account

            # Expand user path if needed
            sa_file = os.path.expanduser(service_account_file)
            if not os.path.exists(sa_file):
                raise FileNotFoundError(f"Service account file not found: {sa_file}")

            creds = service_account.Credentials.from_service_account_file(
                sa_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            client = genai.Client(
                credentials=creds,
                project=project_id,
                location=location,
                vertexai=True,
            )
        else:
            # Use default credentials
            client = genai.Client(project=project_id, location=location, vertexai=True)

        return client

    except Exception as e:
        # Log helpful error message
        logger.error(
            "AUTHENTICATION ERROR: %s\n"
            "This usually means one of the following:\n"
            "• Service account file is missing or invalid\n"
            "• Project ID is not set or incorrect\n"
            "• Application default credentials are not configured\n"
            "• Missing required permissions for Vertex AI",
            e,
        )
        _log_auth_help()
        raise  # Re-raise the original exception


def _get_video_extension_from_mime(mime_type: str) -> str:
    """Get video file extension from MIME type."""
    mime_to_ext = {
        "video/mp4": "mp4",
        "video/avi": "avi",
        "video/mkv": "mkv",
        "video/mov": "mov",
        "video/quicktime": "mov",
        "video/webm": "webm",
        "video/wmv": "wmv",
        "video/flv": "flv",
        "video/3gpp": "3gp",
        "video/mpeg": "mpg",
        "video/x-msvideo": "avi",
    }
    return mime_to_ext.get(mime_type.lower(), "mp4")  # Default to mp4


def _generate_output_paths(
    generated_videos,
    output: str | None = None,
    *,
    prefix: str = "generated_video_",
    directory_name: str = "generated_video_",
    extension_fallback: str = "mp4",
) -> list[str]:
    """
    Generate output file paths for video saving, ensuring no conflicts with existing files.

    Args:
        generated_videos: List of GeneratedVideo objects
        output: Output path specification (None, extension, directory, or full path)
        prefix: Prefix for auto-generated filenames
        directory_name: Name to use when output is directory
        extension_fallback: Default extension if MIME type can't be determined

    Returns:
        List of output file paths that don't conflict with existing files
    """
    num_videos = len(generated_videos)
    is_single_video = num_videos == 1
    output_paths = []

    if output is None:
        # Auto-generate temp files
        for i, video in enumerate(generated_videos):
            # Get extension from MIME type if available (handle both new and legacy API)
            if (
                hasattr(video, "video")
                and video.video
                and hasattr(video.video, "mime_type")
            ):
                mime_type = video.video.mime_type
            else:
                mime_type = getattr(video, "mime_type", f"video/{extension_fallback}")
            ext = _get_video_extension_from_mime(mime_type)

            if is_single_video:
                temp_fd, output_path = tempfile.mkstemp(suffix=f".{ext}", prefix=prefix)
            else:
                temp_fd, output_path = tempfile.mkstemp(
                    suffix=f".{ext}", prefix=f"{prefix}{i:02d}_"
                )
            os.close(temp_fd)  # Close the file descriptor since we'll write separately
            output_paths.append(output_path)

    elif output.startswith("."):
        # Extension provided
        ext = output[1:]  # Remove the leading dot

        # Validate it's a video extension
        video_extensions = {
            "mp4",
            "avi",
            "mkv",
            "mov",
            "wmv",
            "flv",
            "webm",
            "m4v",
            "3gp",
            "mpg",
            "mpeg",
        }
        if ext.lower() not in video_extensions:
            raise ValueError(
                f"Invalid video extension: {ext}. Must be one of: {video_extensions}"
            )

        for i in range(num_videos):
            if is_single_video:
                temp_fd, output_path = tempfile.mkstemp(suffix=f".{ext}", prefix=prefix)
            else:
                temp_fd, output_path = tempfile.mkstemp(
                    suffix=f".{ext}", prefix=f"{prefix}{i:02d}_"
                )
            os.close(temp_fd)
            output_paths.append(output_path)

    elif os.path.isdir(output):
        # Directory provided - generate safe filenames
        # Get list of existing files in the directory for collision detection
        try:
            existing_files = set(os.listdir(output))
        except OSError:
            existing_files = set()

        for i, video in enumerate(generated_videos):
            # Get extension from MIME type if available (handle both new and legacy API)
            if (
                hasattr(video, "video")
                and video.video
                and hasattr(video.video, "mime_type")
            ):
                mime_type = video.video.mime_type
            else:
                mime_type = getattr(video, "mime_type", f"video/{extension_fallback}")
            ext = _get_video_extension_from_mime(mime_type)

            if is_single_video:
                base_filename = f"{directory_name}.{ext}"
            else:
                base_filename = f"{directory_name}_{i:02d}.{ext}"

            # Use non_colliding_key to ensure we don't overwrite existing files
            safe_filename = non_colliding_key(base_filename, existing_files)
            output_path = os.path.join(output, safe_filename)
            output_paths.append(output_path)

            # Add the new filename to existing_files to avoid collisions within this batch
            existing_files.add(safe_filename)

    else:
        # Full path provided
        if is_single_video:
            # For single video, check if file exists and use non_colliding_key if needed
            if os.path.exists(output):
                directory = os.path.dirname(output)
                filename = os.path.basename(output)
                try:
                    existing_files = (
                        set(os.listdir(directory))
                        if directory
                        else set(os.listdir("."))
                    )
                except OSError:
                    existing_files = set()
                safe_filename = non_colliding_key(filename, existing_files)
                output_path = (
                    os.path.join(directory, safe_filename)
                    if directory
                    else safe_filename
                )
            else:
                output_path = output
            output_paths.append(output_path)
        else:
            # Multiple videos - generate indexed paths using non_colliding_key
            base_path, ext = os.path.splitext(output)
            if not ext:
                ext = f".{extension_fallback}"

            directory = os.path.dirname(output)
            try:
                existing_files = (
                    set(os.listdir(directory)) if directory else set(os.listdir("."))
                )
            except OSError:
                existing_files = set()

            for i in range(num_videos):
                # Generate base indexed filename
                base_filename = os.path.basename(f"{base_path}__{i:02d}{ext}")
                # Use non_colliding_key to ensure uniqueness
                safe_filename = non_colliding_key(base_filename, existing_files)
                output_path = (
                    os.path.join(directory, safe_filename)
                    if directory
                    else safe_filename
                )
                output_paths.append(output_path)

                # Add to existing_files to avoid collisions within this batch
                existing_files.add(safe_filename)

    return output_paths


def save_generated_videos(
    video_input,  # Can be: GeneratedVideo, list[GeneratedVideo], operation.response, or operation
    output: str | None = None,
    *,
    prefix: str = "generated_video_",
    directory_name: str = "generated_video_",
    extension_fallback: str = "mp4",
) -> str | list[str]:
    """
    Save generated video(s) to file(s). Handles single videos, lists of videos,
    operation responses, or full operations.

    Args:
        video_input: Can be one of:
            - A single google.genai.types.GeneratedVideo object
            - A list of GeneratedVideo objects (from op.response.generated_videos)
            - An operation response object (op.response)
            - A full operation object (op)
        output: Output path specification. Can be:
            - None: Auto-generate temp file(s) with appropriate extension
            - Path starting with '.': Use as extension (e.g., '.mp4')
            - Full filepath: Use as-is (for single video) or add index for multiple
            - Directory path: Use directory with generated names
        prefix: Prefix for auto-generated filenames (keyword-only)
        directory_name: Name to use when output is directory (keyword-only)
        extension_fallback: Default extension if MIME type can't be determined (keyword-only)

    Returns:
        Single path (str) for one video, or list of paths for multiple videos

    Examples:
        >>> # Save single video to temp file
        >>> path = save_generated_videos(video_obj)  # doctest: +SKIP
        >>> # Save multiple videos with index
        >>> paths = save_generated_videos(op.response.generated_videos)  # doctest: +SKIP
        >>> # Save from full operation
        >>> paths = save_generated_videos(op, '/path/to/videos/')  # doctest: +SKIP
        >>> # Save with specific extension
        >>> path = save_generated_videos(video_obj, '.mp4')  # doctest: +SKIP
    """
    # Extract list of GeneratedVideo objects from various input types
    generated_videos = []

    # Check for operation object first (most specific)
    if hasattr(video_input, "response") and hasattr(
        video_input.response, "generated_videos"
    ):
        # It's a full operation (op)
        generated_videos = list(video_input.response.generated_videos)
    elif hasattr(video_input, "generated_videos"):
        # It's an operation response (op.response)
        generated_videos = list(video_input.generated_videos)
    elif hasattr(video_input, "video") and video_input.video:
        # It's a single GeneratedVideo object (new API format)
        generated_videos = [video_input]
    elif hasattr(video_input, "video_bytes") or hasattr(video_input, "uri"):
        # It's a single GeneratedVideo object (legacy format)
        generated_videos = [video_input]
    elif hasattr(video_input, "__iter__") and not isinstance(video_input, str):
        # It's a list or iterable of videos (check this last to avoid catching operation objects)
        video_list = list(video_input)
        # Verify that the first item is actually a video object
        if video_list and (
            hasattr(video_list[0], "video")
            or hasattr(video_list[0], "video_bytes")
            or hasattr(video_list[0], "uri")
        ):
            generated_videos = video_list
        else:
            raise ValueError(
                f"Iterable contains invalid video objects. First item type: {type(video_list[0]) if video_list else 'empty'}, "
                f"attributes: {[attr for attr in dir(video_list[0]) if not attr.startswith('_')] if video_list else 'none'}"
            )
    else:
        raise ValueError(
            "video_input must be a GeneratedVideo object, list of GeneratedVideo objects, "
            "operation response, or full operation object"
        )

    if not generated_videos:
        raise ValueError("No videos found in the input")

    # Helper function to save a single video
    def _save_single_video(generated_video, output_path: str) -> str:
        # Handle both new and legacy API formats
        video_bytes = None
        mime_type = None

        # New API format: video data is nested in generated_video.video
        if hasattr(generated_video, "video") and generated_video.video:
            video_obj = generated_video.video
            if hasattr(video_obj, "video_bytes") and video_obj.video_bytes:
                video_bytes = video_obj.video_bytes
                mime_type = getattr(
                    video_obj, "mime_type", f"video/{extension_fallback}"
                )
            elif hasattr(video_obj, "uri") and video_obj.uri:
                # Handle case where video is stored as URI (would need to download)
                raise NotImplementedError(
                    "URI-based videos not yet supported. Video should have video_bytes."
                )

        # Legacy API format: video data is directly on generated_video
        elif hasattr(generated_video, "video_bytes") and generated_video.video_bytes:
            if isinstance(generated_video.video_bytes, str):
                # Base64 encoded string
                video_bytes = base64.b64decode(generated_video.video_bytes)
            else:
                # Already bytes
                video_bytes = generated_video.video_bytes
            mime_type = getattr(
                generated_video, "mime_type", f"video/{extension_fallback}"
            )
        elif hasattr(generated_video, "uri") and generated_video.uri:
            # Handle case where video is stored as URI (would need to download)
            raise NotImplementedError(
                "URI-based videos not yet supported. Video should have video_bytes."
            )

        # Check if we successfully got video data
        if video_bytes is None:
            # Provide helpful error message with what we found
            available_attrs = [
                attr for attr in dir(generated_video) if not attr.startswith("_")
            ]
            if hasattr(generated_video, "video"):
                video_attrs = [
                    attr
                    for attr in dir(generated_video.video)
                    if not attr.startswith("_")
                ]
                error_msg = (
                    f"GeneratedVideo object doesn't contain video data. "
                    f"Available attributes: {available_attrs}. "
                    f"Video sub-object attributes: {video_attrs}"
                )
            else:
                error_msg = (
                    f"GeneratedVideo object must have either video_bytes or uri attribute. "
                    f"Available attributes: {available_attrs}"
                )
            raise ValueError(error_msg)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Write video bytes to file
        with open(output_path, "wb") as f:
            f.write(video_bytes)

        return output_path

    # Determine how many videos we're saving
    num_videos = len(generated_videos)
    is_single_video = num_videos == 1

    # Generate output paths using helper function
    output_paths = _generate_output_paths(
        generated_videos,
        output,
        prefix=prefix,
        directory_name=directory_name,
        extension_fallback=extension_fallback,
    )

    # Save all videos
    saved_paths = []
    for video, output_path in zip(generated_videos, output_paths):
        saved_path = _save_single_video(video, output_path)
        saved_paths.append(saved_path)

    # Return single path for single video, list for multiple
    if is_single_video:
        return saved_paths[0]
    else:
        return saved_paths


# Backward compatibility alias
save_generated_video = save_generated_videos


def _get_frame_path(media_path: str, frame_idx: int = 0) -> str:
    """
    Get frame path from media file. If it's a video, extract frame to temp file.
    If it's already an image, return the original path.
    """
    if _is_video_file(media_path):
        from .video_ops import save_frame

        return save_frame(media_path, frame_idx, output="/TMP")
    else:
        return media_path


def _read_image_as_base64(path: str):
    """Read a local image file and return (data: str, mime_type: str)."""
    with open(path, "rb") as f:
        b = f.read()
    data_b64 = base64.b64encode(b).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/png"
    return data_b64, mime_type


def generate_video(
    prompt: str,
    first_frame: str | None = None,
    last_frame: str | None = None,
    *,
    output: Callable | str | bool = save_generated_videos,
    model: str = "veo-2.0-generate-001",  # TODO: manage models better (when aix offers model routing tools)
    aspect_ratio: str = "16:9",
    duration_seconds: int = 5,
    output_gcs_uri: str | None = None,
    service_account_file: str | None = None,
    project_id: str | None = None,
    location: str = "us-central1",
):
    """
    Generate a video using Veo (Vertex AI) and get the file path(s).

    This function provides a complete workflow: generate video(s) and get downloadable file(s).
    By default, videos are automatically saved to temporary files and you get the path(s).

    The ``output`` parameter follows the canonical :mod:`mixing.egress` protocol
    (a path/dir writes there; a callable is a sink applied to the result), with
    two generator-specific sentinels noted below.

    Args:
        prompt: Text prompt for video generation
        first_frame: Path to image/video file for first frame
        last_frame: Path to image/video file for last frame
        output: Egress target / path specification. Controls how generated videos are processed:
            - save_generated_videos (default): Auto-save to temp files, return path(s)
            - "/path/to/file.mp4": Save to specific file path (via mixing.egress)
            - "/path/to/directory/": Save to directory with auto-generated names
            - ".mp4": Save to temp files with specific extension
            - False: Return raw operation without saving (sentinel)
            - lambda x: x or custom callable: Sink applied to the operation
        model: Model to use for generation
        aspect_ratio: Video aspect ratio
        duration_seconds: Video duration
        output_gcs_uri: GCS URI for output
        service_account_file: Path to service account JSON file
        project_id: Google Cloud project ID
        location: Google Cloud location

    Returns:
        Depends on output parameter:
        - Default: File path(s) where video(s) were saved
        - output=False: Raw operation object
        - Custom function: Whatever your function returns

    Examples:
        Basic usage (generate and get file path):
        >>> path = generate_video("A serene forest at dawn")  # doctest: +SKIP
        >>> print(f"Video saved to: {path}")  # doctest: +SKIP

        Save to specific location:
        >>> path = generate_video("Forest scene", output="/my/video.mp4")  # doctest: +SKIP

        Save to directory:
        >>> path = generate_video("Forest scene", output="/output/dir/")  # doctest: +SKIP

        Just get the operation (no saving):
        >>> op = generate_video("Forest scene", output=False)  # doctest: +SKIP
        >>> # Process op.response.generated_videos yourself

        Custom processing:
        >>> def my_saver(op):
        ...     return save_generated_videos(op, prefix="custom_", extension_fallback="webm")
        >>> paths = generate_video("Forest scene", output=my_saver)  # doctest: +SKIP

    Environment Variables:
        VEO_SERVICE_ACCOUNT_FILE or GOOGLE_APPLICATION_CREDENTIALS: Service account JSON path
        VEO_PROJECT_ID, GOOGLE_CLOUD_PROJECT, or GCLOUD_PROJECT: Project ID
        VEO_LOCATION or GOOGLE_CLOUD_LOCATION: Cloud location (default: us-central1)

    Authentication Setup:
        For automatic authentication, set these environment variables:
        ```bash
        export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
        export GOOGLE_CLOUD_PROJECT="your-project-id"
        ```

    Raises:
        Authentication errors with helpful setup instructions
    """
    # TODO: manage models better (when aix offers model routing tools)
    types = require_package("google.genai.types")
    GenerateVideosConfig, Image = types.GenerateVideosConfig, types.Image
    from mixing.egress import is_path_output, is_sink

    # Resolve ``output`` into an egress sink (a callable applied to the
    # operation) or the ``False`` sentinel (return the raw operation).
    if output:
        if is_path_output(output):
            # A path/dir (str): route through save_generated_videos, which
            # itself delegates filename resolution to mixing.egress semantics.
            output_path = output
            output = lambda v: save_generated_videos(v, output_path)
        elif not is_sink(output):
            raise ValueError(
                "output must be a path, a result-consuming callable, or False"
            )

    # Get authentication info from environment if not provided
    env_sa_file, env_project_id, env_location = _get_auth_from_env()

    # Use provided parameters or fall back to environment
    final_sa_file = service_account_file or env_sa_file
    final_project_id = project_id or env_project_id
    final_location = location if location != "us-central1" else env_location

    # Set up client with error handling
    try:
        client = _setup_genai_client(final_sa_file, final_project_id, final_location)
    except Exception as e:
        # Additional context for missing project ID
        if "project" in str(e).lower() and not final_project_id:
            logger.error(
                "MISSING PROJECT ID: no project ID was provided via parameter or "
                "environment variable. Please set one of the following:\n"
                "• export GOOGLE_CLOUD_PROJECT='your-project-id'\n"
                "• export VEO_PROJECT_ID='your-project-id'\n"
                "• Or pass project_id='your-project-id' to the function"
            )
        raise

    # Prepare Image objects if frames provided
    image_obj = None
    last_image_obj = None

    if first_frame:
        # Get last frame (-1) if the video, otherwise use as-is
        frame_path = _get_frame_path(first_frame, frame_idx=-1)
        b64, mime = _read_image_as_base64(frame_path)
        image_obj = Image(
            mime_type=mime,
            image_bytes=b64,
        )
    if last_frame:
        # Get first frame (0) if the video to be the last frame of the generated video
        frame_path = _get_frame_path(last_frame, frame_idx=0)
        b642, mime2 = _read_image_as_base64(frame_path)
        last_image_obj = Image(
            mime_type=mime2,
            image_bytes=b642,
        )

    # Build config
    config = GenerateVideosConfig(
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
        output_gcs_uri=output_gcs_uri,
        last_frame=last_image_obj,
    )

    # Build request with error handling
    try:
        operation = client.models.generate_videos(
            model=model,
            prompt=prompt,
            image=image_obj,
            config=config,
        )
    except Exception as e:
        # Handle API authentication/permission errors
        error_str = str(e).lower()
        if any(
            term in error_str
            for term in ["unauthenticated", "unauthorized", "401", "403"]
        ):
            logger.error(
                "API AUTHENTICATION/PERMISSION ERROR: %s\n"
                "This usually means:\n"
                "• Your authentication credentials are invalid or expired\n"
                "• Your project doesn't have Vertex AI API enabled\n"
                "• Your service account lacks necessary permissions\n"
                "• You need to enable the Vertex AI API in Google Cloud Console\n"
                "Required permissions:\n"
                "• aiplatform.predictions.predict\n"
                "• aiplatform.operations.get",
                e,
            )
            _log_auth_help()
        raise

    # Poll until done with error handling
    try:
        while not operation.done:
            time.sleep(5)
            operation = client.operations.get(operation)
    except Exception as e:
        logger.error(
            "VIDEO GENERATION ERROR: %s\n"
            "This could be due to:\n"
            "• API quota limits exceeded\n"
            "• Invalid model parameters\n"
            "• Network connectivity issues\n"
            "• Service temporarily unavailable",
            e,
        )
        raise

    # Process the result through the resolved ``output`` egress sink. The
    # ``False`` sentinel falls through to returning the raw operation.
    if output:
        try:
            return output(operation)
        except Exception as e:
            logger.error("Error saving video: %s", e)
            from xdol import save_obj

            # TODO: Change this to saving the operation to a temp file and including the path to it in the error message.
            _error_save_path = save_obj(operation)
            logger.error(
                "Saved the operation object as a pickle under: %s", _error_save_path
            )
            raise
    else:
        return operation
