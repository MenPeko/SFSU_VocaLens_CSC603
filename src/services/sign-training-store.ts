import {mkdir, readFile, readdir, writeFile} from "node:fs/promises"
import {join} from "node:path"
import {CORE_SIGN_LABELS, CORE_SIGN_SAMPLE_TARGET, type CoreSignLabel} from "../sign/core-signs"
import {SIGN_SESSION_ROLES, type SignRecordingSession, type SignTrainingSample, type SignTrainingStats} from "../sign/types"

const DATA_DIR = process.env.SIGN_TRAINING_DATA_DIR || "./data/sign-samples"
const SESSION_ID_PATTERN = /^[A-Za-z0-9._-]+$/

export class SignTrainingSampleValidationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "SignTrainingSampleValidationError"
  }
}

function getActiveSessionId(): string | null {
  const value = process.env.SIGN_TRAINING_SESSION?.trim()
  return value ? value : null
}

function getSessionDir(sessionId: string): string {
  return join(DATA_DIR, "sessions", sessionId)
}

// Sessions are created by `python -m sign_classifier_service.eval.sessions new`.
// The store never creates one, so a typo in SIGN_TRAINING_SESSION cannot start
// an unlabeled session that the evaluation would have to guess a role for.
async function loadActiveSession(): Promise<SignRecordingSession | null> {
  const sessionId = getActiveSessionId()
  if (!sessionId) {
    return null
  }

  if (!SESSION_ID_PATTERN.test(sessionId)) {
    throw new SignTrainingSampleValidationError(`SIGN_TRAINING_SESSION=${sessionId} is not a valid session id.`)
  }

  const sessionPath = join(getSessionDir(sessionId), "session.json")
  let session: SignRecordingSession
  try {
    session = JSON.parse(await readFile(sessionPath, "utf8")) as SignRecordingSession
  } catch {
    throw new SignTrainingSampleValidationError(
      `SIGN_TRAINING_SESSION=${sessionId} has no session.json. Create it with sign_classifier_service.eval.sessions new.`,
    )
  }

  if (session.sessionId !== sessionId || !SIGN_SESSION_ROLES.includes(session.role) || !session.signer) {
    throw new SignTrainingSampleValidationError(`${sessionPath} is missing sessionId, role, or signer.`)
  }

  return session
}

function getSamplesRoot(session: SignRecordingSession | null): string {
  return session ? getSessionDir(session.sessionId) : DATA_DIR
}

async function ensureDataDirs(root: string): Promise<void> {
  await mkdir(root, {recursive: true})
  await Promise.all(CORE_SIGN_LABELS.map((label) => mkdir(join(root, label), {recursive: true})))
}

export async function getSignTrainingStats(): Promise<SignTrainingStats> {
  const root = getSamplesRoot(await loadActiveSession())
  await ensureDataDirs(root)

  const labels = {} as Record<CoreSignLabel, number>
  let totalSamples = 0

  for (const label of CORE_SIGN_LABELS) {
    const files = await readdir(join(root, label)).catch(() => [])
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

  const session = await loadActiveSession()
  const root = getSamplesRoot(session)
  await ensureDataDirs(root)

  const storedSample: SignTrainingSample = session
    ? {...sample, sessionId: session.sessionId, signer: session.signer, sessionRole: session.role}
    : sample

  const sampleId = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
  const sampleBasePath = join(root, sample.label, sampleId)

  await writeFile(`${sampleBasePath}.json`, JSON.stringify(storedSample, null, 2), "utf8")

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
