# VocaLens Revised Roadmap

CSC 603/803, Fall 2026

Team: Calvin Phetsoumphou, Juan Navarro Jr, Prag Gupta

Revision date: November 3, 2026

This roadmap revises the approved VocaLens proposal. It keeps the original goals and dates. It updates the plan to build on the existing VocaLens app, use the iPhone 13 and the Fujifilm X-T30, and use public data to improve accuracy.

## Purpose

VocaLens is a hands-free visual assistant. A person points a camera, asks a spoken question, and hears a short answer. If the picture is too unclear to answer, the app says so.

The team will extend the existing VocaLens application instead of starting a new server. The phone remains the camera a person holds. An X-T30 on a tripod is the lab camera for demos and repeatable tests. Public data improves two separate pieces: the spoken visual answers, and the eight-sign classifier already in the repo.

## What stays from the approved proposal

The success targets are unchanged.

- A spoken answer in a median of 5 seconds, with a 90th percentile of 8 seconds or less.
- Speech recognition word error rate of 10% or lower on 100 recorded questions.
- At least 60% human-judged correctness on 200 VizWiz questions, and an "unclear" response on at least 50% of the questions VizWiz marks unanswerable.
- Usability testing with at least 8 people, including at least 2 screen-reader users, and a System Usability Scale score of 70 or higher.
- The whole phone flow operable with VoiceOver or TalkBack, checked against WCAG 2.2 AA.

The deliverables are unchanged: a deployed demo, a public repository, evaluation scripts and results, a written report, and a live presentation.

## Implementation choices

The running app already captures a camera frame, sends it to Gemini, speaks the result, and keeps a session history. The voice question loop builds on that path.

- The phone interface is a press-and-hold control in the existing React webview. It records the question and takes one frame from the iPhone 13.
- The lab camera is the X-T30, powered with a CP-W126 dummy battery and an AC adapter, connected by Micro-HDMI into a USB capture card. The app selects it like any other video device. HDMI supplies a 1080p frame, which is the frame the model sees.
- Speech recognition uses the Whisper API, with faster-whisper available locally. Speech output uses browser text-to-speech, with an adjustable rate.
- Visual answers stay on the Gemini client already in the app. In Week 4 the team scores Gemini against one other hosted vision model on 30 training images. Gemini remains the demo model unless the other model is clearly better on both accuracy and latency.
- The sign classifier stays a separate mode. It recognizes eight signs: wait, help, yes, no, stop, again, thank you, and please.

## How public data is used

**VizWiz** improves the visual answers. The training split is for finding errors and editing the prompt. Errors are grouped into invented text, the wrong object, an answer given for an unreadable photo, and a refusal when the answer is visible. The prompt is revised from those patterns. A 200-question validation sample is graded once, after the prompt is frozen. Those images stay on local machines and out of the repository.

**ASL Citizen** improves the sign classifier. The team keeps only clips whose English gloss is one of the eight signs, runs them through the same MediaPipe extractor the app uses, and trains one model on local recordings and one model on local recordings plus those clips. Both are scored on recordings from the team's own camera that were held out of training. MS-ASL fills a gloss only when ASL Citizen does not contain it. Static fingerspelling collections do not match this model, so they are not used.

VizWiz and ASL Citizen are reported separately. Improvement on one does not count as improvement on the other.

## Roles

- **Calvin Phetsoumphou:** phone interface, camera selection, speech playback and rate, and the screen-reader pass.
- **Juan Navarro Jr:** speech-to-text, the visual-question prompt, session context, latency logging, and the ASL Citizen extraction and training comparison.
- **Prag Gupta:** VizWiz harness and grading, the speech test set, user testing, and the report.

Everyone contributes to the slides and the demo.

## Roadmap

### Week 3, November 2–8: Spoken question to spoken answer

Add press-and-hold capture on the phone. Transcribe the question, send the phone frame and the question to Gemini, and speak the answer. Log a timestamp for capture, transcription, the vision call, and speech. The first prompt asks for a short answer, reads visible text word for word, and says when the image is unclear. The sign mode stays unchanged.

Exit criterion: one live question about a phone photo comes back as speech, with stage timings saved.

### Week 4, November 9–15: Accuracy loop and lab camera

Run the training split of VizWiz through the pipeline. Revise the prompt from the error groups, and allow one follow-up question to reuse the same frame and conversation context. Compare Gemini with one other hosted vision model on 30 training items. Bring up the X-T30 so a frame from the capture card enters the same pipeline. Download ASL Citizen and filter it to the eight glosses.

Exit criterion: a revised prompt with a before-and-after score on a small training subset, a Fuji frame answered out loud, and a filtered sign-clip list.

### Week 5, November 16–22: Latency, access, and the sign model

Resize images before upload and stream transcription and speech where that cuts delay. Make the phone page usable with a screen reader, and add speech-rate control. Record the 100-question speech set. Extract landmarks for the filtered ASL Citizen clips, train the two classifiers, and score them on the held-out local recordings.

Exit criterion: a latency table, a VoiceOver walkthrough, and a sign-model comparison on the team's own camera.

### Week 6, November 23–29: Frozen evaluation

This is the lighter Thanksgiving week, so the prompt and the sign model stay frozen. Two teammates independently grade the 200 VizWiz validation questions. Compute word error rate on the 100 recordings. Deploy the demo. Fix only failures that block the demo or the grading.

Exit criterion: the latency, word-error, VizWiz, and sign-comparison numbers that will appear in the report.

### Week 7, November 30–December 6: User testing

Eight or more participants each complete five scripted tasks, then fill out the System Usability Scale and give open comments. At least two participants use a screen reader. Recruit through the SFSU Disability Programs and Resource Center. Fix only problems that block those tasks.

Exit criterion: a SUS score and a short list of what participants could not complete.

### Week 8, December 7 through the presentation: Report and demo

Finish the report and slides. Rehearse the live demo on the iPhone. Keep the X-T30 as the lab camera for a second shot when the room or the object is hard to frame by hand. Show the sign classifier only with the held-out comparison from Week 5.

## Risks

- **Three model calls can miss the 5-second target.** Each stage is timed from Week 3, images are resized, and the slowest stage is the one that gets optimized in Week 5.
- **The model can answer confidently when the photo is bad.** The prompt has to admit uncertainty, and the VizWiz unanswerable items are a scored test of that behavior. The demo is described as an aid, not a tool for medication or street safety.
- **Public video can fail on the team's camera.** The sign model is accepted only if accuracy rises on held-out recordings from the team's own camera.
- **The X-T30 setup can slip.** The phone path is the graded demo. The Fuji is added after that path already speaks an answer.
- **User recruitment can slip.** DPRC contact is already planned. If blind or low-vision participants are unavailable, sighted participants can complete blindfolded tasks, and the report will state that limit.
- **Hosted APIs can cost money or rate-limit the evaluation.** Evaluation outputs are cached, development calls are capped, and the local speech fallbacks stay available.

## Equipment

- iPhone 13 for the participant-facing demo.
- X-T30, CP-W126 dummy battery, AC adapter, Micro-HDMI cable, and a USB capture card for the lab camera.
- A laptop to run the app, Whisper, Gemini, and the local evaluations.
