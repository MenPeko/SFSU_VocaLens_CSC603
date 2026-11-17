# VocaLens roadmap

CSC 603/803, Fall 2026

Team: Calvin Phetsoumphou, Juan Navarro Jr, Prag Gupta

Revision date: November 3, 2026

This roadmap revises the approved VocaLens proposal. It keeps the original goals and dates. It updates the plan to build on the existing VocaLens app, use the iPhone 13 and the Fujifilm X-T30, and use public data to improve accuracy.

## How a coding agent should use this file

Read this section before changing code. Implement **one week at a time**, and only the week named in the prompt. A prompt such as "implement Week 3 per docs/ROADMAP.md" means Week 3 only. Later weeks are context for names and dates. Do not start them, scaffold them, or leave placeholder results for them.

If the prompt does not name a week, stop after reading this file and ask which week to implement.

### People tasks are not code

These items are done by the team. Do not write stubs, fake scores, sample recordings, recruitment emails, or a report draft for them.

- Grading VizWiz questions by hand, including the Week 6 double grade of 200 validation questions and any human before-and-after score.
- Recording the 100-question speech set.
- User testing, the System Usability Scale, and open comments.
- Outreach to the SFSU Disability Programs and Resource Center.
- The written report, slides, and live presentation.

A coding week may add a harness that *stores* human grades or timings. It must not invent those grades or timings.

### Data and secrets stay out of the repository

- Do not commit VizWiz images, ASL Citizen clips, MS-ASL or Kaggle video, or local sign recordings.
- Keep that data on the machine that runs the evaluation. `data/vizwiz/`, `data/asl-citizen/`, `data/ms-asl/`, and `data/kaggle/` are listed in `.gitignore`.
- Put API keys in `.env`. That file is already listed in `.gitignore`. The Gemini client reads `GEMINI_API_KEY` from the environment. Do not hard-code a key, copy `.env` into an example file, or commit a key in source, tests, or logs.

### Extend the running app

The proposal described a new FastAPI backend. Do not create one. VocaLens question answering extends the Written Text path that already captures a frame, calls Gemini, speaks the result, and pushes the result into session history.

| Step | File | What to extend |
| --- | --- | --- |
| Frame capture | `src/app/CaptionsApp.ts` | `runCaptureAndReadFlow` takes one photo with `session.camera.requestPhoto` after a short glasses-button press. |
| Gemini call | `src/services/gemini-ocr.ts` | `extractTextFromImageWithGemini` sends the image and a prompt. It already does model discovery, retries, and API-error translation. A visual question uses this client. |
| Speech output | `src/app/CaptionsApp.ts` | `session.audio.speak` speaks the result. `src/app/active-session.ts` (`speakOnActiveSession`) is the same speak path for other callers. |
| Session history | `src/api/sse-stream.ts`, `src/webview/App.tsx` | The capture flow broadcasts `status`, `ocr_result`, and `translation_result`. The React webview renders that history. The phone control is a press-and-hold in this webview, not a new server. |

The phone page is the Written Text mode of `src/webview/App.tsx` (`AppMode` `"whiteboard"`). Speech recognition is new: the roadmap uses the Whisper API, with faster-whisper available locally. Browser text-to-speech, with an adjustable rate, is the phone playback path. Glasses playback stays on `session.audio.speak`.

There is no Whisper call in the repo yet. Add it on this path. Do not add a VocaLens route under `sign_classifier_service/`.

### Open confirmation

**Update, November 5, 2026:** the team confirmed the sign-classifier evaluation is in scope. That work follows `docs/SIGN_EVAL_PLAN.md`: held-out test sets, leak checks, ASL Citizen extraction with the app's own extractor, and the baseline-plus-one-dataset loop. Additional datasets (MS-ASL, Kaggle) may be added one at a time. Each one is added only after it passes these checks: it is isolated-sign video (no static fingerspelling images), it has a gloss mapping in `gloss_map.json`, it has signer IDs or a documented workaround, it passes a duplicate check against existing data, and a kept or dropped decision is made from accuracy on the owner's own held-out test session. The course instructor approved any datasets for this academic, non-commercial project. Raw data, landmarks, and weights still stay out of the repository. The full rule is in `docs/DATA_LICENSES.md` (Decisions). Live Sign behavior and the promoted model in `data/sign-models/active-model.json` are unchanged by that plan. The rest of this note still applies to any other sign-mode change.

**Sign mode is out of scope until the team says otherwise.** The roadmap below still describes an eight-sign classifier and ASL Citizen, because that is the team plan. Those paragraphs are not a coding assignment.

Leave these alone:

- `src/webview/components/SignLanguageCapture.tsx`
- `src/webview/hooks/useSignLanguage.ts`
- `src/services/gemini-sign-interpreter.ts`
- `src/services/sign-classifier-client.ts`
- `src/sign/`
- `sign_classifier_service/`
- `touchdesigner/`

Do not filter ASL Citizen, extract landmarks, train a classifier, or change Live Sign mode while this note stands. Weeks 4, 5, 6, and 8 mention that work. Skip those bullets until a prompt confirms the sign classifier is in scope.

### Decisions already made

Do not reopen these. They replace the new-server design in the original proposal.

- The phone a person holds is an iPhone 13. The lab camera is a Fujifilm X-T30 on a tripod, powered by a CP-W126 dummy battery and an AC adapter, connected by Micro-HDMI to a USB capture card. The app selects that card like any other video device. The X-T30 comes after the phone path already speaks an answer.
- Visual answers stay on the Gemini client in `src/services/gemini-ocr.ts`. In Week 4 the team may score one other hosted vision model on 30 training images. Gemini remains the demo model unless that model is clearly better on both accuracy and latency.
- The first prompt asks for a short answer, reads visible text word for word, and says when the image is unclear.
- VizWiz and ASL Citizen are separate. Improvement on one does not count as improvement on the other. Static fingerspelling collections are not used. Other sign datasets (MS-ASL, Kaggle) are added one at a time, each only after it passes the checks in `docs/DATA_LICENSES.md` (Decisions).

## Purpose

VocaLens is a hands-free visual assistant. A person points a camera, asks a spoken question, and hears a short answer. If the picture is too unclear to answer, the app says so.

The team will extend the existing VocaLens application instead of starting a new server. The phone remains the camera a person holds. An X-T30 on a tripod is the lab camera for demos and repeatable tests. Public data improves two separate pieces: the spoken visual answers, and the eight-sign classifier already in the repo. The sign-classifier piece is on hold for coding agents until the team confirms it. See [Open confirmation](#open-confirmation).

## What stays from the approved proposal

The success targets are unchanged.

- A spoken answer in a median of 5 seconds, with a 90th percentile of 8 seconds or less.
- Speech recognition word error rate of 10% or lower on 100 recorded questions.
- At least 60% human-judged correctness on 200 VizWiz questions, and an "unclear" response on at least 50% of the questions VizWiz marks unanswerable.
- Usability testing with at least 8 people, including at least 2 screen-reader users, and a System Usability Scale score of 70 or higher.
- The whole phone flow operable with VoiceOver or TalkBack, checked against WCAG 2.2 AA.

The deliverables are unchanged: a deployed demo, a public repository, evaluation scripts and results, a written report, and a live presentation.

## Implementation choices

The running app already captures a camera frame, sends it to Gemini, speaks the result, and keeps a session history. The voice question loop builds on that path. File names are in [Extend the running app](#extend-the-running-app).

- The phone interface is a press-and-hold control in the existing React webview. It records the question and takes one frame from the iPhone 13.
- The lab camera is the X-T30, powered with a CP-W126 dummy battery and an AC adapter, connected by Micro-HDMI into a USB capture card. The app selects it like any other video device. HDMI supplies a 1080p frame, which is the frame the model sees.
- Speech recognition uses the Whisper API, with faster-whisper available locally. Speech output uses browser text-to-speech, with an adjustable rate. Glasses speech stays on `session.audio.speak`.
- Visual answers stay on the Gemini client already in the app. In Week 4 the team scores Gemini against one other hosted vision model on 30 training images. Gemini remains the demo model unless the other model is clearly better on both accuracy and latency.
- The sign classifier stays a separate mode. It recognizes eight signs: wait, help, yes, no, stop, again, thank you, and please. Coding agents do not change this mode until the team confirms it is in scope.

## How public data is used

**VizWiz** improves the visual answers. The training split is for finding errors and editing the prompt. Errors are grouped into invented text, the wrong object, an answer given for an unreadable photo, and a refusal when the answer is visible. The prompt is revised from those patterns. A 200-question validation sample is graded once, after the prompt is frozen. Those images stay on local machines and out of the repository. Hand grading is a people task.

**ASL Citizen** improves the sign classifier. This dataset work is on hold with the rest of the sign mode. When the team confirms it, the plan is: keep only clips whose English gloss is one of the eight signs, run them through the same MediaPipe extractor the app uses, and train one model on local recordings and one model on local recordings plus those clips. Both are scored on recordings from the team's own camera that were held out of training. Other datasets (MS-ASL, Kaggle) may each be added after ASL Citizen, one at a time, under the checks in `docs/DATA_LICENSES.md` (Decisions). Static fingerspelling collections do not match this model, so they are not used.

VizWiz and ASL Citizen are reported separately. Improvement on one does not count as improvement on the other.

## Roles

- **Calvin Phetsoumphou:** phone interface, camera selection, speech playback and rate, and the screen-reader pass.
- **Juan Navarro Jr:** speech-to-text, the visual-question prompt, session context, latency logging, and the ASL Citizen extraction and training comparison.
- **Prag Gupta:** VizWiz harness and grading, the speech test set, user testing, and the report.

Everyone contributes to the slides and the demo. The report, slides, grading, recordings, and user testing are people tasks.

## Roadmap

Each week ends with an exit criterion. A coding agent implements only the week it was asked for, and only the bullets marked as code.

### Week 3, November 2–8: Spoken question to spoken answer

**Code.** Add press-and-hold capture on the phone. Transcribe the question, send the phone frame and the question to Gemini, and speak the answer. Log a timestamp for capture, transcription, the vision call, and speech. The first prompt asks for a short answer, reads visible text word for word, and says when the image is unclear. The sign mode stays unchanged.

**People.** None.

Exit criterion: one live question about a phone photo comes back as speech, with stage timings saved.

The task list and acceptance checks for this week are in [Week 3 task brief](#week-3-task-brief).

### Week 4, November 9–15: Accuracy loop and lab camera

**Code.** Run the training split of VizWiz through the pipeline. Revise the prompt from error groups the team supplies. Allow one follow-up question to reuse the same frame and conversation context. Compare Gemini with one other hosted vision model on 30 training items. Bring up the X-T30 so a frame from the capture card enters the same pipeline.

**Code, only after the sign classifier is confirmed.** Download ASL Citizen and filter it to the eight glosses. Until then, skip this bullet.

**People.** Hand grading and the before-and-after judgment on the small training subset. Do not invent those scores.

Exit criterion: a revised prompt with a before-and-after score on a small training subset, a Fuji frame answered out loud, and a filtered sign-clip list. The filtered list waits on the sign-classifier confirmation.

### Week 5, November 16–22: Latency, access, and the sign model

**Code.** Resize images before upload and stream transcription and speech where that cuts delay. Make the phone page usable with a screen reader, and add speech-rate control.

**Code, only after the sign classifier is confirmed.** Extract landmarks for the filtered ASL Citizen clips, train the two classifiers, and score them on the held-out local recordings.

**People.** Record the 100-question speech set. Do not synthesize stand-in audio or a word-error number.

Exit criterion: a latency table, a VoiceOver walkthrough, and a sign-model comparison on the team's own camera. The sign-model comparison waits on the sign-classifier confirmation.

### Week 6, November 23–29: Frozen evaluation

This is the lighter Thanksgiving week, so the prompt and the sign model stay frozen.

**Code.** Deploy the demo. Fix only failures that block the demo or the grading. Cache evaluation outputs. Do not retune the prompt.

**People.** Two teammates independently grade the 200 VizWiz validation questions. Compute word error rate only from the real 100 recordings, once those recordings exist. Do not fill in latency, word-error, VizWiz, or sign-comparison numbers.

Exit criterion: the latency, word-error, VizWiz, and sign-comparison numbers that will appear in the report.

### Week 7, November 30–December 6: User testing

**Code.** Fix only problems that block the scripted tasks, and only after the team reports those problems.

**People.** Eight or more participants each complete five scripted tasks, then fill out the System Usability Scale and give open comments. At least two participants use a screen reader. Recruit through the SFSU Disability Programs and Resource Center.

Exit criterion: a SUS score and a short list of what participants could not complete. Both come from the sessions, not from code.

### Week 8, December 7 through the presentation: Report and demo

**Code.** None, unless a demo-blocking fix is named in the prompt.

**People.** Finish the report and slides. Rehearse the live demo on the iPhone. Keep the X-T30 as the lab camera for a second shot when the room or the object is hard to frame by hand. Show the sign classifier only with the held-out comparison from Week 5, and only if that comparison was confirmed and run.

## Risks

- **Three model calls can miss the 5-second target.** Each stage is timed from Week 3, images are resized, and the slowest stage is the one that gets optimized in Week 5.
- **The model can answer confidently when the photo is bad.** The prompt has to admit uncertainty, and the VizWiz unanswerable items are a scored test of that behavior. The demo is described as an aid, not a tool for medication or street safety.
- **Public video can fail on the team's camera.** The sign model is accepted only if accuracy rises on held-out recordings from the team's own camera. Coding agents do not start this work while the sign classifier is unconfirmed.
- **The X-T30 setup can slip.** The phone path is the graded demo. The Fuji is added after that path already speaks an answer.
- **User recruitment can slip.** DPRC contact is already planned. If blind or low-vision participants are unavailable, sighted participants can complete blindfolded tasks, and the report will state that limit. Recruitment and that report note are people tasks.
- **Hosted APIs can cost money or rate-limit the evaluation.** Evaluation outputs are cached, development calls are capped, and the local speech fallbacks stay available.

## Equipment

- iPhone 13 for the participant-facing demo.
- X-T30, CP-W126 dummy battery, AC adapter, Micro-HDMI cable, and a USB capture card for the lab camera.
- A laptop to run the app, Whisper, Gemini, and the local evaluations.

## Week 3 task brief

Implement this section only when the prompt asks for Week 3. Do not start Week 4.

### Goal

One live spoken question about a photo from the phone comes back as speech, and the app saves a timestamp for each stage.

### In scope

1. Add a press-and-hold control to Written Text mode in `src/webview/App.tsx`. Holding it records the question and takes one frame from the phone camera. Releasing it starts the pipeline.
2. Transcribe that recording with the Whisper API. If the API is unavailable, fall back to local faster-whisper. Read the key from `.env`. Do not commit the key.
3. Send that frame and the transcript to Gemini through `src/services/gemini-ocr.ts`, using a prompt that asks for a short answer, reads visible text word for word, and says the image is unclear when it cannot be answered.
4. Speak the answer in the browser with text-to-speech. Keep `session.audio.speak` in `src/app/CaptionsApp.ts` working for the existing glasses button flow.
5. Show the question and answer in the existing session history (`src/api/sse-stream.ts` and the history panel in `src/webview/App.tsx`).
6. Save a timestamp for capture, transcription, the vision call, and speech. The existing Written Text flow in `runCaptureAndReadFlow` is the pattern for those stages.

### Out of scope

- Live Sign, the eight-sign classifier, ASL Citizen, MS-ASL, and every path listed under [Open confirmation](#open-confirmation).
- The X-T30 and the capture card.
- VizWiz downloads, prompt revision from error groups, a second vision model, and follow-up questions. Those are Week 4.
- Image resizing and streaming. Those are Week 5.
- A new FastAPI service. Do not add VocaLens routes to `sign_classifier_service/`.
- Any file under `touchdesigner/`.
- Placeholder scores, recordings, or report text.

### Acceptance checks

- A person can press and hold on the phone page, ask one question about what the camera sees, and hear a spoken answer.
- The saved timings include capture, transcription, the vision call, and speech.
- An image the model cannot read produces an answer that says it is unclear, rather than a guessed description.
- Visible text in the frame is returned as written, not paraphrased.
- The glasses-button Written Text flow still captures a photo, calls Gemini, and speaks.
- Live Sign mode still builds and is unchanged.
- `.env` is untracked. No API key, VizWiz image, or sign clip is added to the commit.
