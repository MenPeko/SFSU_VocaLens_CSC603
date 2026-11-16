export interface VocalensTurn {
  role: "user" | "model"
  text: string
}

export interface VocalensSession {
  id: string
  image: Buffer
  mimeType: string
  turns: VocalensTurn[]
  updatedAt: number
  captureAt: number | null
}

export const VOCALENS_SESSION_IDLE_MS = 30 * 60 * 1000

const sessions = new Map<string, VocalensSession>()

export function clearAllVocalensSessions(): void {
  sessions.clear()
}

export function replaceVocalensSession(
  id: string,
  image: Buffer,
  mimeType: string,
  captureAt: number | null,
  now = Date.now(),
): VocalensSession {
  const session: VocalensSession = {
    id,
    image: Buffer.from(image),
    mimeType: mimeType || "image/jpeg",
    turns: [],
    updatedAt: now,
    captureAt,
  }
  sessions.set(id, session)
  return session
}

export function getVocalensSession(id: string, now = Date.now()): VocalensSession | null {
  const session = sessions.get(id)
  if (!session) return null
  if (now - session.updatedAt > VOCALENS_SESSION_IDLE_MS) {
    sessions.delete(id)
    return null
  }
  return session
}

export function appendVocalensTurn(id: string, turn: VocalensTurn, now = Date.now()): void {
  const session = sessions.get(id)
  if (!session) {
    throw new Error("Session not found")
  }
  session.turns.push(turn)
  session.updatedAt = now
}

export function clearVocalensSession(id: string): void {
  sessions.delete(id)
}

export function dropIdleVocalensSessions(now = Date.now()): void {
  for (const [id, session] of sessions) {
    if (now - session.updatedAt > VOCALENS_SESSION_IDLE_MS) {
      sessions.delete(id)
    }
  }
}
