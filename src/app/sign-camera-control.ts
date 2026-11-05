export interface SignCameraSettings {
  enabled: boolean
}

type SignCameraListener = (settings: SignCameraSettings) => void

let signCameraEnabled = false
const listeners = new Set<SignCameraListener>()

function getCurrentSettings(): SignCameraSettings {
  return {enabled: signCameraEnabled}
}

export function getSignCameraSettings(): SignCameraSettings {
  return getCurrentSettings()
}

export function isSignCameraEnabled(): boolean {
  return signCameraEnabled
}

export function setSignCameraEnabled(enabled: boolean): SignCameraSettings {
  if (signCameraEnabled === enabled) {
    return getCurrentSettings()
  }

  signCameraEnabled = enabled
  const nextSettings = getCurrentSettings()

  for (const listener of listeners) {
    listener(nextSettings)
  }

  return nextSettings
}

export function subscribeToSignCameraSettings(listener: SignCameraListener): () => void {
  listeners.add(listener)

  return () => {
    listeners.delete(listener)
  }
}
