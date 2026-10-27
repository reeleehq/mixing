"""
Video tools

"""

from mixing.util import VIDEO_IMPORT_NAMES, require_media_extra

require_media_extra("mixing.video", modules=VIDEO_IMPORT_NAMES)

from mixing.video.video_subtitles import (
    generate_subtitle_clips,
    to_srt_time,
    write_subtitles_in_video,
)
from mixing.video.video_concat import (
    concatenate_videos,
    crossfade_overlap,
    crossfade_transition,
    fade_through_black,
    max_overlap_for_clips,
    needs_crossfade_overlap,
    overlap_blend,
    slow_motion_blend,
    trim_and_crossfade,
    trim_first_frame_from_subsequent_clips,
)
from mixing.video.thumbnail import (
    grab_frame,
    make_thumbnail,
    THUMBNAIL_SIZE,
    YOUTUBE_THUMB_SIZE,
)
from mixing.video.gif import make_gif
from mixing.video.video_ops import (
    Video,
    VideoFrames,
    crop_video,
    save_frame,
    loop_video,
    replace_audio,
    normalize_audio,
    overlay_ambient_bed,
    DEFAULT_AMBIENT_MIX_RATIO,
    change_speed,
    ken_burns_video,
    ken_burns_film,
    assemble_audio_track,
)
from mixing.video.video_util import (
    SOCIAL_SIZES,
    get_video_dimensions,
    resize_to_dimensions,
    normalize_video_dimensions,
)

from mixing.video.shot_boundaries import (
    SHOT_STRATEGIES,
    shots,
    contact_sheet,
    adaptive_threshold,
    frame_changes,
    spans_from_changes,
    merge_short_spans,
)
from mixing.video.cutaways import (
    Cutaway,
    MediaInfo,
    CutawayPlan,
    cutaways,
    plan_cutaways,
    probe_media,
    render_cutaways,
)
