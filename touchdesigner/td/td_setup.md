# TouchDesigner patch setup

This file only covers the TD side. Start the Python process first (see
`touchdesigner/README.md`).

## 1. Minimum setup (receive everything)

1. Add an **OSC In CHOP** (`oscin1`).
   - **Network Port:** `7400` (must match `--osc-port`).
   - **Split Bundles into Individual Messages:** Off
   - **Queued:** Off (live channels — no latency penalty).
2. That's it — every `/vision/...` address becomes a CHOP channel whose
   name is the address path. Rename channels with a Rename CHOP if you
   want shorter names.

Channels you'll commonly drive visuals with:

| CHOP channel                              | Typical use                        |
|-------------------------------------------|------------------------------------|
| `vision/hand/left/wrist_x`                | X position of left hand            |
| `vision/hand/left/pinch_state`            | Trigger (0/1)                      |
| `vision/hand/left/pinch_distance`         | Continuous pinch amount            |
| `vision/hand/left/openness`               | Fist-to-open control               |
| `vision/hand/left/gesture`                | DAT-only (string)                  |
| `vision/session/event/started`            | One-frame pulse to trigger scenes  |

## 2. Catching the `gesture` string

`OSC In CHOP` drops string payloads. Pair it with an **OSC In DAT**
listening on the same port, with the same address filter:

- **Active:** On
- **Network Port:** `7400`
- **Address Wildcard:** `/vision/hand/*/gesture`

The DAT's last row has the most recent gesture per side — feed it to a
Script CHOP or a Select DAT if you want to react to specific labels
(e.g. trigger a particle burst on `victory`).

## 3. Triggering one-shots on session end

`/vision/session/event/ended` sends `1` for a single frame. The classic TD
pattern:

    [OSC In CHOP] -> [Select CHOP 'vision/session/event/ended']
                  -> [Trigger CHOP]   # rising-edge detector
                  -> your target op

`session/event/started` works the same way.

## 4. Coordinate systems at a glance

- `raw_wrist_x` / `raw_wrist_y`: **0..1 image space**. Multiply by your
  TOP resolution to overlay a dot exactly where MediaPipe sees the wrist.
- `wrist_x` / `wrist_y` / `wrist_z`: **shoulder-centered units**. `(0, 0)`
  is the midpoint between shoulders; the shoulder width is `1`. Stable
  across subjects and camera distances — prefer these for gestural
  control mapped to scene-space.

## 5. Classifier output (optional)

If the Python process is launched with
`--classifier-model /path/to/artifact.json`, it will run the existing
temporal sign classifier on every detected session and publish:

- `/vision/classifier/status` (string)
- `/vision/classifier/label` (string)
- `/vision/classifier/confidence` (float 0..1)

Route the string addresses through an **OSC In DAT** (same setup as for
the gesture string).
