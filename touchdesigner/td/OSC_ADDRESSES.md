# OSC address schema

All addresses are sent to the configured OSC host/port (default
`127.0.0.1:7400`) as soon as each frame is processed. Every address carries
exactly one value so an `OSC In CHOP` turns each address into a channel 1:1.

Root prefix: `/vision` (configurable via `config.OSC_ROOT`).

## Heartbeat

| Address                            | Type    | Description |
|------------------------------------|---------|-------------|
| `/vision/heartbeat/frame_index`    | int     | Monotonic counter, use as a "new frame" trigger |
| `/vision/heartbeat/fps`            | float   | EMA-smoothed FPS                                |
| `/vision/heartbeat/dropped_frames` | int     | Cumulative camera-read failures                 |
| `/vision/heartbeat/timestamp_ms`   | float   | Monotonic clock millis for the current frame    |

## Pose

| Address                          | Type  | Range / Unit                       |
|----------------------------------|-------|------------------------------------|
| `/vision/pose/shoulder_center_x` | float | 0..1 image coords                  |
| `/vision/pose/shoulder_center_y` | float | 0..1 image coords                  |
| `/vision/pose/shoulder_width`    | float | 0..~0.5 image-space distance       |

`shoulder_width` doubles as a cheap proxy for distance-from-camera (smaller
= farther away). Everything else in `/hand/*` is normalized against it.

## Hands (`left` / `right`)

All hand coordinates are in the same SHOULDER-CENTERED, SHOULDER-NORMALIZED
frame used by the original trained classifier. `raw_wrist_*` is the
unnormalized 0..1 image-space coordinate, useful for driving TOP overlays.

| Address                              | Type   | Description                                      |
|--------------------------------------|--------|--------------------------------------------------|
| `/vision/hand/<side>/present`        | int    | 0 / 1                                            |
| `/vision/hand/<side>/openness`       | float  | 0..1 (0 = fist, 1 = fully open)                  |
| `/vision/hand/<side>/palm_facing`    | float  | 0..1 (1 = palm faces camera)                     |
| `/vision/hand/<side>/extended_count` | float  | 0..1 (fraction of fingers extended)              |
| `/vision/hand/<side>/pinch_distance` | float  | 0..~0.4 (thumb-tip to index-tip, shoulder units) |
| `/vision/hand/<side>/pinch_state`    | int    | 0 / 1 (Schmitt-triggered)                        |
| `/vision/hand/<side>/wrist_x`        | float  | Shoulder-centered x                              |
| `/vision/hand/<side>/wrist_y`        | float  | Shoulder-centered y                              |
| `/vision/hand/<side>/wrist_z`        | float  | Shoulder-centered z                              |
| `/vision/hand/<side>/raw_wrist_x`    | float  | 0..1 image-space x                               |
| `/vision/hand/<side>/raw_wrist_y`    | float  | 0..1 image-space y                               |
| `/vision/hand/<side>/wrist_vx`       | float  | shoulder units / s                               |
| `/vision/hand/<side>/wrist_vy`       | float  | shoulder units / s                               |
| `/vision/hand/<side>/gesture_score`  | float  | 0..1                                             |
| `/vision/hand/<side>/gesture`        | string | `none` \| `open_palm` \| `fist` \| `pointing` \| `victory` \| `thumbs_up` |

## Face

| Address                     | Type  | Description |
|-----------------------------|-------|-------------|
| `/vision/face/present`      | int   | 0 / 1       |
| `/vision/face/mouth_open`   | float | 0..1        |
| `/vision/face/nose_x`       | float | shoulder-centered |
| `/vision/face/nose_y`       | float | shoulder-centered |

## Session (hands-up / hands-down detector)

| Address                          | Type  | Description |
|----------------------------------|-------|-------------|
| `/vision/session/active`         | int   | 1 while either hand is visible (+ small debounce) |
| `/vision/session/duration_ms`    | int   | How long the current session has been active |
| `/vision/session/event/started`  | int   | Fires value=1 on exactly one frame when a session opens |
| `/vision/session/event/ended`    | int   | Fires value=1 on one frame when the hands-down timer trips AND minimum-duration gate passes |

Use `event/started` / `event/ended` to trigger DAT scripts or sample-and-hold.

## Classifier (optional, only fires on session end)

Only present when `--classifier-model` is configured.

| Address                        | Type    | Description                                       |
|--------------------------------|---------|---------------------------------------------------|
| `/vision/classifier/status`    | string  | `recognized` / `uncertain` / `no_sign`            |
| `/vision/classifier/label`     | string  | One of the CORE_SIGN_LABELS, or `none`            |
| `/vision/classifier/confidence`| float   | 0..1                                              |
