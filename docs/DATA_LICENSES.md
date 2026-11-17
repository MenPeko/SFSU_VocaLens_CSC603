# Dataset licenses

Read on November 4, 2026. This note records clauses from the pages below. It is not a legal opinion, and it does not add permissions the text does not state.

No dataset file was downloaded. `data/asl-citizen/` and `data/ms-asl/` are not in this working copy, and neither directory contains a LICENSE, README, or terms file.

Planned use, from `docs/ROADMAP.md`: filter clips to eight glosses (wait, help, yes, no, stop, again, thank you, please), extract MediaPipe landmarks, train two classifiers, and score them on held-out recordings from the team's camera. The course repository is public. The dataset is not redistributed.

## ASL Citizen

### Source

- Project page: https://www.microsoft.com/en-us/research/project/asl-citizen/
- Dataset license: https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/
- Recommended use (separate from the license): https://www.microsoft.com/en-us/research/project/asl-citizen/recommended-use/
- Download page: https://www.microsoft.com/en-us/download/details.aspx?id=105253
- Code repository, separate from the dataset: https://github.com/microsoft/ASL-citizen-code

The download page lists `ASL_Citizen.zip` (42.8 GB) and `Demographics.zip` (5.0 KB). The project page also prints a direct `wget` URL for `ASL_Citizen.zip`. Those files were not downloaded.

### License name

Microsoft Research License Terms, on the dataset-license page. The project page says: "Dataset License: Please see the supporting tab."

The code repository has its own MIT license. That license was not used as the dataset license. The dataset-license page says the terms apply to "the source code, object code, or data (collectively 'Materials') that accompany this license."

### a. Is academic or non-commercial research use allowed?

The license allows non-commercial, non-revenue-generating research use. It does not use the word "academic."

> Subject to the terms of this agreement, you have the below rights, if applicable, to use the Materials solely for non-commercial, non-revenue generating, research purposes

The capstone described in `docs/ROADMAP.md` is research and is not described as revenue-generating. The same section still requires every other term, including the personal-data and consent sentences below.

The project page sends commercial use somewhere else: "If you are interested in commercial use, please contact ASL_Citizen@microsoft.com."

Data use must also match the original consent. The consent form was not on the pages read, so what that consent allows is **UNCLEAR**.

> you may use and modify the data, but your use and modification must be consistent with the consent under which the data was provided and/or gathered and you may not distribute the data or your modifications to the data.

### b. Can we publish code that downloads, filters, and processes the dataset?

**UNCLEAR** for a script the team writes. The license never mentions downstream code. It does forbid distributing the dataset and modifications of the data.

> you may not distribute the data or your modifications to the data.

> share, publish, distribute or lend the Materials, provide the Materials as a stand-alone hosted solution for others to use, or transfer the Materials or this agreement to any third party.

A public repository that contains the zip, the clips, `Demographics.zip`, or modified data is distribution of the Materials under those sentences. The license does not say whether a script that only points at Microsoft's download URL is itself distribution.

Section 1(a) limits source code that is part of the Materials ("you may not distribute the source code"). That sentence is about source code included with the license, not about this course repository. The download page lists two zip files and does not say they contain source code.

### c. Can we commit derived data (landmarks, trained weights, evaluation outputs)?

Clips and other unmodified dataset files: no. See the two quotes in (b).

Landmarks, weights, and evaluation files: **UNCLEAR**, because the license uses "modifications to the data" and "results" and does not define either word, and it does not define "material or substantial portion."

Modifications cannot be distributed:

> you may not distribute the data or your modifications to the data.

Results may be published only without a material or substantial portion of the Materials:

> You may publish (or present papers or articles) on your results from using the Materials provided that no material or substantial portion of the Materials is included in any such publication or presentation.

The text does not say whether MediaPipe landmarks, a filtered manifest, model weights, or an evaluation dump are modifications or results. Until that is explicit, those files stay on the machine that runs the evaluation and out of the public repository.

### d. Can we show clips or frames in a class presentation or report?

**UNCLEAR** for a single frame, because two phrases are undefined: "third party" and "material or substantial portion."

Identifiable people are personal data, and personal data is not to be shared:

> If the data (set forth in Section 1(c) above) includes or is found to include any data that enables any ability to identify an individual ("Personal Data"), you will not use such Personal Data for any purpose other than was authorized and consented to by the data subject/research participant. You will not use Personal Data to contact any person. You will keep Personal Data in strict confidence. You will not share any Personal Data that is collected or in your possession with any third party for any reason and as required under the original consent agreement. Further, you will destroy the Personal Data and any backup or copies, immediately upon the completion of your research.

A presentation may cover results only if it leaves out a material or substantial portion of the Materials. The publication sentence is quoted in (c). The license does not say whether one frame or one short clip crosses that line.

The project page says the videos were "collected with consent from each contributor under IRB approval." It does not quote that consent.

### e. What citation or attribution is required?

The license text has no citation sentence. It does forbid stripping notices that are already in the Materials:

> remove, minimize, block, or modify any notices of Microsoft or its suppliers in the Materials

The project page, which is not the license, asks for a paper citation:

> Citation: If you use this dataset in your work, please cite our paper

```
@article{desai2023asl,
  title={ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition},
  author={Desai, Aashaka and Berger, Lauren and Minakov, Fyodor O and Milan, Vanessa and Singh, Chinmay and Pumphrey, Kriston and Ladner, Richard E and Daum{\'e} III, Hal and Lu, Alex X and Caselli, Naomi and Bragg, Danielle},
  journal={arXiv preprint arXiv:2304.05934},
  year={2023}
}
```

Whether that request is a condition of the license is **UNCLEAR**. It is stated on the project page and is absent from the license text. The report should still include it, because the publisher asks for it on the same site as the license.

### f. Is there a registration, agreement, or approval step before use?

No registration form, account, or approval step is described for the research use in section 1. The license says use is acceptance:

> BY USING THE MATERIALS, YOU ACCEPT THESE TERMS.

The project page publishes a direct download URL, and the download page says to see the project page for the license. Neither page shows a click-through agreement. Whether the Download Center adds terms that the fetched page did not show is **UNCLEAR**.

Commercial use is a separate contact step: "If you are interested in commercial use, please contact ASL_Citizen@microsoft.com." That sentence is not an approval gate for the non-commercial research use in section 1.

### Recommended-use page

This page is linked from the project site and is not part of the Microsoft Research License Terms. It is written as a request, not as a grant. Whether it binds a user of the license is **UNCLEAR**.

> We ask that this dataset be used with an aim of making the world more equitable and just for deaf people, and with a commitment to do no harm. In that spirit, this dataset should not be used to develop technology that purports to replace sign language interpreters, fluent signing educators, and/or other hard-won accommodations for deaf people. We also ask that users of this dataset make no attempt to identify participants, or to use this dataset for applications that might exploit participant identity or appearance, including (but not limited to) facial recognition, deepfakes, or identification of sensitive attributes like race.

The page also says the dataset "was designed primarily" for isolated sign recognition as dictionary retrieval, and it cautions against using it alone for continuous signing. An eight-gloss isolated classifier is isolated-sign work. Whether a class demo "purports to replace" interpreters is **UNCLEAR** from this paragraph. The roadmap already calls the demo an aid and says to show the sign classifier with the team's own held-out recordings.

### Compliance actions

1. Keep the ASL Citizen use inside non-commercial, non-revenue-generating research. Do not ship a paid product or a hosted service on these clips.
2. Leave `data/asl-citizen/` gitignored. Do not commit `ASL_Citizen.zip`, `Demographics.zip`, clips, frames, or filtered copies of those files.
3. Do not commit landmarks, filtered manifests, or weights trained on this dataset while (c) is UNCLEAR. The ignore lines in [Lines to add](#lines-to-add) cover the folders this repo already writes.
4. Do not put dataset clips or frames in the report, the slides, or the live demo. Use the team's own camera recordings for anything a person can see.
5. Put the `desai2023asl` citation in the report and anywhere the dataset is named.
6. Do not strip Microsoft notices from files that come with the download.
7. Delete the videos, demographics, landmarks derived from them, and backups when the class research is finished. The personal-data section says to destroy personal data and copies "immediately upon the completion of your research."
8. Read the contributor consent before use if the team can get it. The license requires use "consistent with the consent," and that consent was not on the pages read.
9. In the report, describe the sign demo as an aid. Do not describe it as a replacement for an interpreter.

## MS-ASL

### Source

- Project page: https://www.microsoft.com/en-us/research/project/ms-asl/
- Publication page: https://www.microsoft.com/en-us/research/publication/ms-asl-a-large-scale-data-set-and-benchmark-for-understanding-american-sign-language/
- Download page: https://www.microsoft.com/en-us/download/details.aspx?id=100121 (`MS-ASL.zip`, 1.9 MB, version August 2019, date published 7/15/2024)

`data/ms-asl/` does not exist, so there is no local LICENSE or README.

### License name

**Not found.** The project page, the publication page, and the download-details page do not name a license or paste terms. The project page says only: "The data set is publicly available to the community." That sentence does not state conditions.

`MS-ASL.zip` was not downloaded. The agreement, if it is inside that zip, was not read.

Unofficial pages repeat a README that says the package is "Licensed under the Computational Use of Data Agreement (C-UDA)" and names `C-UDA-0.1_annotated_discussion.pdf`. Those pages are not Microsoft's dataset page. The PDF was not read. Microsoft also publishes a different document, Computational Use of Data Agreement v1.0. That v1.0 text was not applied here, because nothing on the official MS-ASL pages says the dataset uses it.

### a–f

Each answer is **UNCLEAR**. The terms were not on the official pages, not in this working copy, and not read from the zip.

| Question | Status |
| --- | --- |
| a. Academic or non-commercial research use | **UNCLEAR.** No use grant was found. |
| b. Publish code that downloads, filters, and processes the dataset | **UNCLEAR.** No clause about downstream code or redistribution was found. |
| c. Commit landmarks, weights, or evaluation outputs | **UNCLEAR.** No clause about modifications or results was found. |
| d. Show clips or frames in class | **UNCLEAR.** No clause about publication or personal data was found. |
| e. Citation or attribution | **UNCLEAR.** No attribution requirement was found. The publication page names Hamid Vaezi Joze and Oscar Koller, BMVC, September 2019, and the title "MS-ASL: A Large-Scale Data Set and Benchmark for Understanding American Sign Language." It does not say to cite that paper. |
| f. Registration, agreement, or approval | **UNCLEAR.** The download page says "Click Download and follow the instructions." It does not show an agreement. The fetched download page returned no license text. |

The course instructor approved the use of any datasets for this project (see [Decisions](#decisions)). The clauses above stay **UNCLEAR** until someone reads the license file that accompanies the official download and quotes it here.

### Compliance actions

1. MS-ASL may be added under the additional-dataset checks in [Decisions](#decisions), one dataset at a time. When the package is downloaded, quote the agreement that ships with it in this section.
2. Keep `data/ms-asl/` gitignored, which the root `.gitignore` already does.
3. If a gloss is missing from ASL Citizen, record the gap in the report unless a dataset that passed the checks and was kept covers it.

## Repository audit

Checked November 4, 2026. No files were changed except this note.

### Raw data ignore rules

Root `.gitignore` already contains:

```
# Local evaluation data. Never commit VizWiz images, ASL Citizen clips, or MS-ASL video.
data/vizwiz/
data/asl-citizen/
data/ms-asl/
```

`data/asl-citizen/`, `data/ms-asl/`, and `data/vizwiz/` are not present on disk.

### Where landmarks and weights are written

The trainer and reporter already use these paths (`sign_classifier_service/README.md` and `sign_classifier_service/train.py`):

| Path | What lands there | Ignore rule today |
| --- | --- | --- |
| `data/sign-samples/` | Landmark JSON plus RGB and skeleton images. `--manifest` can also point at a JSON or JSONL file anywhere. | `data/sign-samples/.gitignore` ignores `*.json`, `*.jpg`, and `*.jpeg` only. |
| `data/sign-exports/` | JSONL export of landmark sequences. | `data/sign-exports/.gitignore` is `*` with an exception for `.gitignore`. |
| `data/sign-models/` | Versioned model JSON, a metrics JSON, and `active-model.json`. | `data/sign-models/.gitignore` is `*` with exceptions for `.gitignore` and `active-model.json`. |
| `data/sign-reports/` | `report.md`, `report.json`, `full-eval.json`, and related evaluation files. | No `.gitignore`. |
| `data/sign-samples-archive/` | Older landmark JSON and JPG copies (`wait-wrong-20260420`: 50 JSON, 100 JPG). | No `.gitignore`. |

`active-model.json` is a pointer. It names `data/sign-models/core-sign-20260414T210536Z.json` and does not contain weights. The weight file itself matches the `*` rule in that folder.

### Gaps

1. `data/sign-samples/.gitignore` does not ignore `*.png`. Six skeleton PNGs are under `data/sign-samples/wait/`. A dataset frame saved as PNG in that tree would not match the current rules. MP4, MOV, WEBM, NPY, and NPZ are also unmatched.
2. `data/sign-reports/` is the evaluation-output directory and is not ignored. A report that included dataset frames, landmarks, or clip identifiers would be committable.
3. `data/sign-samples-archive/` holds landmark JSON and JPG files and is not ignored.
4. `--manifest` and `--output-dir` can write outside the folders above. A new directory for ASL Citizen landmarks is not ignored unless it is created under `data/asl-citizen/`.
5. `data/sign-models/active-model.json` is an allow-list exception. It is safe only while it stays a pointer. A future file of the same name that embeds weights would not be ignored.

The JSON files under `data/sign-samples/` that were opened use `"labelSource": "manual_capture"` (372 files). They are local recordings, not ASL Citizen or MS-ASL files. The current model report evaluates 404 local samples. That does not close the ignore gaps above.

### Git history

This working copy has no `.git` directory. There is no commit history and no tracked-file list to search. History was not rewritten. Nothing in the working tree is named as an ASL Citizen or MS-ASL object: those two directories are absent, and the sign files on disk are the local captures and the local model described above.

### Lines to add

Add these lines to the root `.gitignore`, directly under the existing `data/ms-asl/` line:

```
data/sign-reports/
data/sign-samples-archive/
data/landmarks/
data/sign-samples/**/*.png
data/sign-samples/**/*.mp4
data/sign-samples/**/*.mov
data/sign-samples/**/*.webm
data/sign-samples/**/*.npy
data/sign-samples/**/*.npz
```

Add these lines to `data/sign-samples/.gitignore`:

```
*.png
*.mp4
*.mov
*.webm
*.npy
*.npz
```

`data/landmarks/` and `models/sign/` each contain a `README.md` and nothing else. Derived files written there are ignored. If an extractor uses another path, add that path before the first run.

Do not remove the `!active-model.json` exception unless the pointer itself starts carrying weights. Keep the versioned model JSON covered by the existing `*` in `data/sign-models/.gitignore`.

The root `.gitignore` now uses a stricter rule than the extension list above: every file under `data/sign-samples/`, `data/sign-samples-archive/`, `data/sign-reports/`, `data/landmarks/`, and `models/sign/` is ignored except `README.md`. See [Decisions](#decisions).

## Decisions

These are the repository's choices. They do not add permissions or definitions the license text does not state. Where a clause above is **UNCLEAR**, the file stays off the public tree.

- Landmarks, model weights, and evaluation outputs derived from ASL Citizen stay on the machine that produced them. `data/landmarks/` holds landmark files. `models/sign/` holds model weights. `data/sign-reports/` holds evaluation outputs. `data/sign-samples/` and `data/sign-samples-archive/` are ignored on the same rule so a dataset frame or landmark file in those trees is not committed. The only tracked file in each tree is `README.md`.
- Committed summaries hold aggregate numbers only. `docs/SIGN_RESULTS.md`, `reports/dataset-audit.*`, and `reports/baseline-evaluation.*` contain counts, accuracies, and distributions. They contain no clip names, video IDs, signer IDs, session IDs, landmarks, or weights. Any new file under `reports/` follows the same rule.
- Dataset clips and frames are not shown in slides or reports. Anything a person can see uses the team's own camera recordings. The ASL Citizen skeleton image that `sign:eval-extract` writes to `data/sign-reports/` is an extraction check. It stays local under the `data/sign-reports/` rule.
- **Additional datasets (updated November 7, 2026).** Additional datasets, such as MS-ASL and Kaggle sign datasets, may be added one at a time. Each one is added only after it passes all of these checks:
  1. It is isolated-sign video. Static fingerspelling images are not used.
  2. Its glosses are mapped in `sign_classifier_service/eval/gloss_map.json`.
  3. It has signer IDs, or a documented workaround that keeps one signer's clips on one side of every split.
  4. It passes a duplicate check against the data already in use.
  5. It is kept or dropped based on accuracy on the owner's own held-out test session (`own-heldout`), as in the keep rule in `docs/SIGN_EVAL_PLAN.md`.

  The course instructor approved the use of any datasets for this academic, non-commercial project. Raw data, landmarks, and weights still stay out of the repository, on the same rule as the ASL Citizen bullet above. A gloss that no kept dataset covers is recorded as a gap.
- Cite Desai et al. 2023 wherever the dataset is named. The `desai2023asl` entry is in the ASL Citizen section above.
