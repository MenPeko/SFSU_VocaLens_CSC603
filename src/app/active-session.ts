import type {AppSession} from "@mentra/sdk"

interface ActiveSessionRecord {
  session: AppSession
  sessionId: string
  userId: string
  connectedAt: number
}

const activeSessions = new Map<string, ActiveSessionRecord>()

export function registerActiveSession(sessionId: string, userId: string, session: AppSession): void {
  activeSessions.set(sessionId, {
    session,
    sessionId,
    userId,
    connectedAt: Date.now(),
  })
}

export function unregisterActiveSession(sessionId: string): void {
  activeSessions.delete(sessionId)
}

export function getPrimaryActiveSession(): AppSession | null {
  const latest = [...activeSessions.values()].sort((left, right) => right.connectedAt - left.connectedAt)[0]
  return latest?.session ?? null
}

export async function speakOnActiveSession(text: string, displayText?: string): Promise<boolean> {
  const session = getPrimaryActiveSession()
  if (!session) {
    return false
  }

  if (displayText) {
    session.layouts.showTextWall(displayText)
  }

  await session.audio.speak(text)
  return true
}

export function getActiveSessionCount(): number {
  return activeSessions.size
}
