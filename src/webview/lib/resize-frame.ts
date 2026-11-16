export const MAX_FRAME_EDGE = 1280
export const JPEG_QUALITY = 0.8

export function fitFrameSize(
  width: number,
  height: number,
  maxEdge = MAX_FRAME_EDGE,
): {width: number; height: number} {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return {width: 0, height: 0}
  }

  const longest = Math.max(width, height)
  if (longest <= maxEdge) {
    return {width: Math.round(width), height: Math.round(height)}
  }

  const scale = maxEdge / longest
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  }
}

export async function resizeVideoFrame(video: HTMLVideoElement): Promise<Blob> {
  const {width, height} = fitFrameSize(video.videoWidth, video.videoHeight)
  if (width === 0 || height === 0) {
    throw new Error("The camera frame is not ready yet. Hold the button a moment longer.")
  }

  const canvas = document.createElement("canvas")
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext("2d")
  if (!context) {
    throw new Error("Could not prepare the camera frame.")
  }

  context.drawImage(video, 0, 0, width, height)
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY))
  if (!blob) {
    throw new Error("Could not prepare the camera frame.")
  }

  return blob
}
