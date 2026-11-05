import {mkdir, readdir, writeFile} from "node:fs/promises"
import {join} from "node:path"
import {CORE_SIGN_LABELS, CORE_SIGN_SAMPLE_TARGET, type CoreSignLabel} from "../sign/core-signs"
import type {SignTrainingSample, SignTrainingStats} from "../sign/types"

const DATA_DIR = process.env.SIGN_TRAINING_DATA_DIR || "./data/sign-samples"

export class SignTrainingSampleValidationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "SignTrainingSampleValidationError"
  }
}

function getLabelDir(label: CoreSignLabel): string {
  return join(DATA_DIR, label)
}

async function ensureDataDirs(): Promise<void> {
  await mkdir(DATA_DIR, {recursive: true})
  await Promise.all(CORE_SIGN_LABELS.map((label) => mkdir(getLabelDir(label), {recursive: true})))
}

export async function getSignTrainingStats(): Promise<SignTrainingStats> {
  await ensureDataDirs()

  const labels = {} as Record<CoreSignLabel, number>
  let totalSamples = 0

  for (const label of CORE_SIGN_LABELS) {
    const files = await readdir(getLabelDir(label)).catch(() => [])
    const sampleCount = files.filter((fileName) => fileName.endsWith(".json")).length
    labels[label] = sampleCount
    totalSamples += sampleCount
  }

  return {
    totalSamples,
    sampleTarget: CORE_SIGN_SAMPLE_TARGET,
    labels,
  }
}

export async function saveSignTrainingSample(sample: SignTrainingSample): Promise<{
  sampleId: string
  stats: SignTrainingStats
}> {
  if (sample.labelSource !== "manual_capture") {
    throw new SignTrainingSampleValidationError(
      "Only manual_capture samples can be written to the classifier training dataset.",
    )
  }

  if (!sample.extractorVersion || typeof sample.extractorVersion !== "string" || !sample.extractorVersion.trim()) {
    // A missing extractor version would silently contaminate the training set
    // — refuse loudly rather than persist an untraceable sample. See M1 task 2
    // in DEVELOPMENT_ROADMAP.md.
    throw new SignTrainingSampleValidationError(
      "extractorVersion is required on every captured sample — the client must stamp MEDIAPIPE_EXTRACTOR_VERSION on the payload.",
    )
  }

  await ensureDataDirs()

  const sampleId = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
  const labelDir = getLabelDir(sample.label)
  const sampleBasePath = join(labelDir, sampleId)

  await writeFile(`${sampleBasePath}.json`, JSON.stringify(sample, null, 2), "utf8")

  if (sample.base64Grid?.trim()) {
    await writeFile(`${sampleBasePath}-rgb.jpg`, Buffer.from(sample.base64Grid, "base64"))
  }

  if (sample.base64SkeletonGrid?.trim()) {
    const skeletonExtension = sample.skeletonMimeType === "image/png" ? "png" : "jpg"
    await writeFile(`${sampleBasePath}-skeleton.${skeletonExtension}`, Buffer.from(sample.base64SkeletonGrid, "base64"))
  }

  return {
    sampleId,
    stats: await getSignTrainingStats(),
  }
}
