export interface AudioInputChoice {
  deviceId: string
  label: string
}

export function isUnusableAudioInput(label: string): boolean {
  return /dock|virtual|zoom/i.test(label)
}

export function isBuiltInMicrophone(label: string): boolean {
  return /built-in|macbook|internal microphone/i.test(label)
}

export function pickAudioInputId(devices: AudioInputChoice[], selectedId: string): string {
  if (selectedId && devices.some((device) => device.deviceId === selectedId)) return selectedId

  const labeled = devices.filter((device) => device.deviceId && device.deviceId !== "default")
  const builtIn = labeled.find((device) => isBuiltInMicrophone(device.label))
  if (builtIn) return builtIn.deviceId

  const usable = labeled.find((device) => device.label && !isUnusableAudioInput(device.label))
  return usable?.deviceId || labeled[0]?.deviceId || devices[0]?.deviceId || ""
}
