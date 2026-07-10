# Narration Timeline is the master clock

The final Narration Timeline will be the sole authority for Scene boundaries and total media duration. It is created from normalized, measured narration only after the one allowed Duration Repair opportunity; visual planning, caption delivery, music fitting, and FFmpeg rendering derive their timing from it rather than calculating independent schedules. This delays images and music until narration is stable and gives up some possible stage parallelism, but prevents repaired speech, captions, images, music, and container duration from drifting apart.
