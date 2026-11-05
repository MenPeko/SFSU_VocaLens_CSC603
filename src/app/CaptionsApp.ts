import {AppServer, AppSession, type ManagedStreamStatus} from "@mentra/sdk"
import {extractTextFromImageWithGemini} from "../services/gemini-ocr"
import {translateText} from "../services/gemini-translate"
import {sseManager} from "../api/sse-stream"
import {getTargetLanguage} from "../api/routes"
import {registerActiveSession, unregisterActiveSession} from "./active-session"
import {
  getGlassesCameraStreamState,
  resetGlassesCameraStreamState,
  setGlassesCameraStreamState,
  updateGlassesCameraStreamFromManagedStatus,
} from "./glasses-camera-stream"
import {isSignCameraEnabled, subscribeToSignCameraSettings} from "./sign-camera-control"
import {
  FALLBACK_MANAGED_STREAM_QUALITY,
  PRIMARY_MANAGED_STREAM_QUALITY,
  doesManagedStreamQualityMatch,
  isRetryEligibleManagedStreamError,
} from "./managed-stream-quality"

export class CaptionsApp extends AppServer {
  protected async onSession(session: AppSession, sessionId: string, userId: string): Promise<void> {
    session.logger.info(`New session: ${sessionId} for user ${userId}`)
    registerActiveSession(sessionId, userId, session)

    const broadcastCameraStreamState = (streamState: ReturnType<typeof setGlassesCameraStreamState>) => {
      sseManager.broadcast({
        type: "camera_stream",
        stream: streamState,
      })
    }

    const handleManagedStreamStatus = (status: ManagedStreamStatus) => {
      const streamState = updateGlassesCameraStreamFromManagedStatus(status, sessionId, userId)
      session.logger.info(
        {
          streamStatus: status.status,
          streamId: status.streamId,
          hlsUrl: status.hlsUrl,
          quality: status.quality,
          actualQuality: status.actualQuality,
          message: status.message,
        },
        "Managed glasses stream status update",
      )
      broadcastCameraStreamState(streamState)
    }

    const cleanupManagedStreamStatus = session.camera.onManagedStreamStatus(handleManagedStreamStatus)

    const publishInactiveGlassesStreamState = (message: string) => {
      const inactiveState = setGlassesCameraStreamState({
        status: "idle",
        transport: "managed",
        sessionId,
        userId,
        requestedQuality: undefined,
        actualQuality: undefined,
        qualityFallback: undefined,
        actualInputWidth: undefined,
        actualInputHeight: undefined,
        message,
      })
      broadcastCameraStreamState(inactiveState)
    }

    const startGlassesSignStream = async () => {
      const desiredQuality = PRIMARY_MANAGED_STREAM_QUALITY
      const currentStream = getGlassesCameraStreamState()
      if (
        currentStream.sessionId === sessionId &&
        currentStream.requestedQuality === desiredQuality &&
        (currentStream.status === "starting" ||
          currentStream.status === "initializing" ||
          currentStream.status === "preparing" ||
          currentStream.status === "active")
      ) {
        broadcastCameraStreamState(currentStream)
        return
      }

      const startingState = setGlassesCameraStreamState({
        status: "starting",
        transport: "managed",
        sessionId,
        userId,
        requestedQuality: desiredQuality,
        actualQuality: undefined,
        qualityFallback: false,
        message: "Starting Mentra glasses camera stream...",
      })
      broadcastCameraStreamState(startingState)

      try {
        const existingStream = await session.camera.checkExistingStream()
        if (existingStream.hasActiveStream && existingStream.streamInfo?.type === "unmanaged") {
          const conflictState = setGlassesCameraStreamState({
            status: "error",
            transport: "managed",
            sessionId,
            userId,
            message: "The glasses camera is already in use by another stream.",
          })
          broadcastCameraStreamState(conflictState)
          return
        }

        if (existingStream.hasActiveStream && existingStream.streamInfo?.type === "managed") {
          const managedStreamInfo = existingStream.streamInfo as typeof existingStream.streamInfo & {
            type: "managed"
            quality?: "720p" | "1080p"
            actualQuality?: "720p" | "1080p"
          }
          const existingQuality = managedStreamInfo.actualQuality || managedStreamInfo.quality
          if (doesManagedStreamQualityMatch(existingQuality, desiredQuality)) {
            const existingState = setGlassesCameraStreamState({
              status: managedStreamInfo.status === "active" ? "active" : "starting",
              transport: "managed",
              sessionId,
              userId,
              streamId: managedStreamInfo.streamId,
              playbackUrl: managedStreamInfo.hlsUrl,
              playbackMimeType: "application/vnd.apple.mpegurl",
              hlsUrl: managedStreamInfo.hlsUrl,
              dashUrl: managedStreamInfo.dashUrl,
              webrtcUrl: managedStreamInfo.webrtcUrl,
              previewUrl: managedStreamInfo.previewUrl,
              thumbnailUrl: managedStreamInfo.thumbnailUrl,
              requestedQuality: desiredQuality,
              actualQuality: existingQuality,
              qualityFallback: false,
              message: "Joined existing Mentra glasses stream.",
            })
            broadcastCameraStreamState(existingState)
            return
          }

          session.logger.info(
            {
              desiredQuality,
              existingQuality,
              streamId: managedStreamInfo.streamId,
            },
            "Existing managed stream quality mismatched requested quality; rebuilding stream for this app boot",
          )

          await session.camera.stopManagedStream().catch((stopError) => {
            session.logger.warn({stopError}, "Failed to invalidate mismatched managed stream before restart")
          })
        }

        let managedStream: Awaited<ReturnType<typeof session.camera.startManagedStream>>
        let actualQuality = desiredQuality
        let qualityFallback = false

        try {
          managedStream = await session.camera.startManagedStream({
            quality: desiredQuality,
            enableWebRTC: true,
          })
        } catch (startError) {
          if (!isRetryEligibleManagedStreamError(startError)) {
            throw startError
          }

          session.logger.warn(
            {startError, desiredQuality, fallbackQuality: FALLBACK_MANAGED_STREAM_QUALITY},
            "Managed stream startup failed at 1080p; retrying once at 720p",
          )

          managedStream = await session.camera.startManagedStream({
            quality: FALLBACK_MANAGED_STREAM_QUALITY,
            enableWebRTC: true,
          })
          actualQuality = FALLBACK_MANAGED_STREAM_QUALITY
          qualityFallback = true
        }

        const preparingState = setGlassesCameraStreamState({
          status: "preparing",
          transport: "managed",
          sessionId,
          userId,
          streamId: managedStream.streamId,
          playbackUrl: managedStream.hlsUrl,
          playbackMimeType: "application/vnd.apple.mpegurl",
          hlsUrl: managedStream.hlsUrl,
          dashUrl: managedStream.dashUrl,
          webrtcUrl: managedStream.webrtcUrl,
          previewUrl: managedStream.previewUrl,
          thumbnailUrl: managedStream.thumbnailUrl,
          requestedQuality: desiredQuality,
          actualQuality,
          qualityFallback,
          message: qualityFallback
            ? "Mentra glasses stream connected at 720p after a 1080p startup fallback. Waiting for playback to become available."
            : "Mentra glasses stream connected. Waiting for playback to become available.",
        })
        broadcastCameraStreamState(preparingState)
      } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to start glasses camera stream."
        session.logger.error({error: message}, "Managed glasses stream failed")
        const errorState = setGlassesCameraStreamState({
          status: "error",
          transport: "managed",
          sessionId,
          userId,
          requestedQuality: desiredQuality,
          message,
        })
        broadcastCameraStreamState(errorState)
      }
    }

    const stopGlassesSignStream = async (inactiveMessage: string) => {
      const currentStream = getGlassesCameraStreamState()
      const streamBelongsToSession = !currentStream.sessionId || currentStream.sessionId === sessionId
      const isAlreadyInactive = currentStream.status === "idle" || currentStream.status === "stopped"

      if (streamBelongsToSession && isAlreadyInactive) {
        publishInactiveGlassesStreamState(inactiveMessage)
        return
      }

      const stoppingState = setGlassesCameraStreamState({
        status: "stopping",
        transport: "managed",
        sessionId,
        userId,
        streamId: currentStream.streamId,
        message: inactiveMessage,
      })
      broadcastCameraStreamState(stoppingState)

      try {
        await session.camera.stopManagedStream()
      } catch (error) {
        session.logger.warn({error}, "Failed to stop managed glasses stream")
      }

      publishInactiveGlassesStreamState(inactiveMessage)
    }

    let desiredSignStreamEnabled = isSignCameraEnabled()
    let streamSyncPromise = Promise.resolve()

    const syncGlassesSignStream = () => {
      streamSyncPromise = streamSyncPromise
        .then(async () => {
          while (true) {
            const nextDesiredState = desiredSignStreamEnabled

            if (nextDesiredState) {
              await startGlassesSignStream()
            } else {
              await stopGlassesSignStream("Live Sign camera is off while Written Text mode is active.")
            }

            if (desiredSignStreamEnabled === nextDesiredState) {
              break
            }
          }
        })
        .catch((error) => {
          session.logger.error({error}, "Failed to sync glasses sign stream state")
        })

      return streamSyncPromise
    }

    const requestSignStreamSync = (enabled: boolean) => {
      desiredSignStreamEnabled = enabled
      void syncGlassesSignStream()
    }

    const cleanupSignCameraSettings = subscribeToSignCameraSettings(({enabled}) => {
      session.logger.info({enabled}, "Sign camera setting updated")
      requestSignStreamSync(enabled)
    })

    session.layouts.showTextWall("Whiteboard Reader ready. Press a glasses button to capture and read text.")
    sseManager.broadcast({type: "status", status: "ready", message: "Ready. Press glasses button to capture."})
    publishInactiveGlassesStreamState(
      desiredSignStreamEnabled
        ? "Preparing the Live Sign camera stream..."
        : "Live Sign camera is off while Written Text mode is active.",
    )
    void syncGlassesSignStream()

    const runCaptureAndReadFlow = async () => {
      await stopGlassesSignStream("Pausing live sign camera so the glasses can capture an OCR photo.")

      try {
        sseManager.broadcast({type: "status", status: "capturing", message: "Capturing photo..."})
        session.layouts.showTextWall("Capturing photo...")

        const photo = await session.camera.requestPhoto({
          saveToGallery: false,
          size: "medium",
          compress: "none",
        })

        sseManager.broadcast({type: "status", status: "extracting", message: "Extracting text..."})
        session.layouts.showTextWall("Extracting text...")

        const ocrResult = await extractTextFromImageWithGemini(photo.buffer, photo.mimeType)
        const extractedText = ocrResult.text.trim()

        if (!extractedText || extractedText === "NO_TEXT_FOUND") {
          const noTextMessage = "I could not read text from the whiteboard. Please try again."
          sseManager.broadcast({type: "status", status: "ready", message: noTextMessage})
          session.layouts.showTextWall(noTextMessage)
          await session.audio.speak(noTextMessage)
          return
        }

        sseManager.broadcast({type: "ocr_result", text: extractedText})

        const targetLang = getTargetLanguage()
        sseManager.broadcast({type: "status", status: "translating", message: `Translating to ${targetLang}...`})

        const translation = await translateText(extractedText, targetLang)

        sseManager.broadcast({
          type: "translation_result",
          originalText: translation.originalText,
          translatedText: translation.translatedText,
          detectedLanguage: translation.detectedLanguage,
          targetLanguage: translation.targetLanguage,
          timestamp: Date.now(),
          sourceType: "text",
          sourceLabel: "Whiteboard OCR",
        })

        const displayText = translation.translatedText || extractedText
        const spokenText = displayText.length > 400 ? `${displayText.slice(0, 400)}.` : displayText

        if (translation.detectedLanguage !== "unknown" && translation.detectedLanguage.toLowerCase() !== targetLang.toLowerCase()) {
          session.layouts.showTextWall(`[${translation.detectedLanguage}] ${spokenText}`)
        } else {
          session.layouts.showTextWall(`Read text: ${spokenText}`)
        }

        await session.audio.speak(spokenText)
        sseManager.broadcast({type: "status", status: "ready", message: "Ready. Press glasses button to capture."})
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        session.logger.error({error: message}, "Failed whiteboard read flow")
        session.layouts.showTextWall(`Read failed: ${message}`)
        sseManager.broadcast({type: "error", message})
        sseManager.broadcast({type: "status", status: "ready", message: "Ready. Press glasses button to capture."})
        await session.audio.speak(`Read failed. ${message}`)
      } finally {
        if (desiredSignStreamEnabled) {
          requestSignStreamSync(true)
        }
      }
    }

    session.events.onButtonPress(async (event) => {
      session.logger.info({buttonId: event.buttonId, pressType: event.pressType}, "Button press received")
      if (event.pressType === "short") {
        await runCaptureAndReadFlow()
      }
    })

    session.events.onDisconnected(() => {
      session.logger.info(`Session ${sessionId} disconnected.`)
      cleanupSignCameraSettings()
      cleanupManagedStreamStatus()
      void session.camera.stopManagedStream().catch((error) => {
        session.logger.warn({error}, "Failed to stop managed glasses stream during disconnect")
      })
      unregisterActiveSession(sessionId)
      sseManager.broadcast({type: "status", status: "disconnected", message: "Session disconnected."})
      sseManager.broadcast({
        type: "camera_stream",
        stream: resetGlassesCameraStreamState(sessionId),
      })
    })
  }
}
