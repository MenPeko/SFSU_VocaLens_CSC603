# Recording checklist

Three sessions for the held-out evaluation in `docs/SIGN_EVAL_PLAN.md`. All are signed by `owner` on camera `iphone13`. The session folders and `session.json` files already exist under `data/sign-samples/sessions/`. That folder is gitignored, so the clips stay local.

## Signs

Record all eight in every session: `wait`, `help`, `yes`, `no`, `stop`, `again`, `thank_you`, `please`.

## Sessions

| Role | Session ID | Lighting | Day | Target per sign | Total |
| --- | --- | --- | --- | ---: | ---: |
| train | `2026-11-07_owner_desk-daylight` | desk, daylight | November 7, 2026 | 20 | 160 |
| validation | `2026-11-07_owner_desk-evening` | desk, evening | November 7, 2026 | 10 | 80 |
| test | `2026-11-08_owner_window-daylight` | window, daylight | a different day from training | 10 | 80 |

Start the app with one session active. Stop it (Ctrl+C) before switching to the next session.

```bash
# Train: desk, daylight. 20 clips per sign.
SIGN_TRAINING_SESSION=2026-11-07_owner_desk-daylight bun run dev

# Validation: desk, evening. 10 clips per sign.
SIGN_TRAINING_SESSION=2026-11-07_owner_desk-evening bun run dev

# Test: window, daylight, on a different day. 10 clips per sign.
SIGN_TRAINING_SESSION=2026-11-08_owner_window-daylight bun run dev
```

The test session is dated November 8, 2026. If you record it on another day, delete its folder while it is still empty, create it again with the real date, and update `sign_classifier_service/eval/test_sessions.json`:

```bash
rm -r data/sign-samples/sessions/2026-11-08_owner_window-daylight
.venv/bin/python -m sign_classifier_service.eval.sessions new --role test --signer owner --lighting window-daylight --camera iphone13 --date YYYY-MM-DD
```

## Track progress

Print clip counts per sign for every session:

```bash
bun run sign:eval-sessions
```

## Before each session

- Check the terminal: the app must start with the right `SIGN_TRAINING_SESSION`. If `session.json` is missing, the app refuses to save rather than writing to the wrong place.
- Set up the lighting the session is named for, and keep it the same for the whole session.
- Keep the framing on your shoulders, both hands, and face, as in live use.

## Between takes

- Move a little between takes: shift left or right by a hand's width, and turn slightly toward either side.
- Change distance: alternate between arm's length and about one and a half arm's lengths from the camera.
- Vary camera height a little (chest level to eye level) across the session, not within one take.
- Vary speed and size naturally: some takes slower, some quicker, some smaller.
- Rest your hands at your sides between takes, so each clip starts and ends from a neutral pose.
- Record signs in a mixed order instead of all 20 of one sign in a row.
- Delete a take that is cut off, has a hand out of frame, or is the wrong sign. Do not keep it and fix the label later. The app has no delete button: remove `<sampleId>.json`, `<sampleId>-rgb.jpg`, and `<sampleId>-skeleton.png` (or `.jpg`) from `data/sign-samples/sessions/<sessionId>/<sign>/`.

## After recording

- Do not copy or move clips between session folders. The leak checks treat that as a failure.
- Check `bun run sign:eval-sessions` shows the targets above, then run `bun run sign:eval-run --experiment baseline`.
