import {useEffect, useRef, useState} from "react"
import {pickAudioInputId} from "../lib/audio-input"
import {resizeVideoFrame} from "../lib/resize-frame"
import {
  cancelSpeech,
  MAX_SPEECH_RATE,
  MIN_SPEECH_RATE,
  readSpeechRate,
  speakText,
  writeSpeechRate,
} from "../lib/speech"

interface AskTimings {
  captureAt: number | null
  transcriptionStartedAt: number
  transcriptionEndedAt: number
  visionStartedAt: number
  visionEndedAt: number
}

interface AskEvent {
  type: string
  text?: string
  message?: string
  question?: string
  answer?: string
  timings?: AskTimings
}

function createSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID()
  }
  return `vocalens_${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`
}

function pickAudioMime(): string {
  if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") return ""
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"]
  return types.find((type) => MediaRecorder.isTypeSupported(type)) || ""
}

function startRecorder(stream: MediaStream, onData: (blob: Blob) => void): MediaRecorder {
  const audioTracks = stream.getAudioTracks()
  const audioStream = audioTracks.length > 0 ? new MediaStream(audioTracks) : stream
  const mimeType = pickAudioMime()
  const attempts: Array<MediaRecorderOptions | undefined> = mimeType ? [{mimeType}, undefined] : [undefined]
  let lastError: unknown

  for (const options of attempts) {
    try {
      const recorder = options ? new MediaRecorder(audioStream, options) : new MediaRecorder(audioStream)
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) onData(event.data)
      }
      recorder.start(250)
      return recorder
    } catch (error) {
      lastError = error
    }
  }

  throw lastError instanceof Error ? lastError : new Error("The microphone could not start recording.")
}

const fieldLabelStyle = {
  fontSize: "0.7rem",
  color: "#000000",
  letterSpacing: "0.05em",
  fontWeight: 700,
  display: "block",
  marginBottom: "0.35rem",
} as const

const controlStyle = {
  width: "100%",
  background: "#ffffff",
  color: "#000000",
  border: "4px solid #000000",
  padding: "0.55rem 0.7rem",
  fontSize: "0.85rem",
  fontFamily: "inherit",
  fontWeight: 700,
  boxShadow: "4px 4px 0 #FF6600",
} as const

export function VocalensCapture() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const sessionIdRef = useRef(createSessionId())
  const captureAtRef = useRef<number | null>(null)
  const hasFrameRef = useRef(false)
  const busyRef = useRef(false)
  const recordingRef = useRef(false)
  const armingRef = useRef(false)
  const holdRef = useRef<{current: boolean} | null>(null)
  const audioDeviceIdRef = useRef("")
  const audioChoiceExplicitRef = useRef(false)

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([])
  const [deviceId, setDeviceId] = useState("")
  const [audioDeviceId, setAudioDeviceId] = useState("")
  const [rate, setRate] = useState(readSpeechRate)
  const [hasFrame, setHasFrame] = useState(false)
  const [recording, setRecording] = useState(false)
  const [busy, setBusy] = useState(false)
  const [statusText, setStatusText] = useState("Press and hold to ask about what the camera sees.")
  const [answerText, setAnswerText] = useState("")

  useEffect(() => {
    void refreshDevices()
    return () => {
      const recorder = recorderRef.current
      if (recorder && recorder.state !== "inactive") {
        recorder.onstop = () => undefined
        recorder.stop()
      }
      stopStream()
    }
  }, [])

  function stopStream() {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
  }

  async function refreshDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) return
    const listed = await navigator.mediaDevices.enumerateDevices()
    const cameras = listed.filter((device) => device.kind === "videoinput")
    const microphones = listed.filter((device) => device.kind === "audioinput")
    setDevices(cameras)
    setAudioDevices(microphones)
    setDeviceId((current) => current || cameras[0]?.deviceId || "")
    const picked = pickAudioInputId(
      microphones.filter((device) => device.label),
      audioDeviceIdRef.current,
    )
    if (picked && !audioDeviceIdRef.current) {
      audioDeviceIdRef.current = picked
      setAudioDeviceId(picked)
    }
  }

  function audioConstraints(): MediaTrackConstraints | boolean {
    const selected = audioDeviceIdRef.current
    if (!selected || selected === "default") {
      return {echoCancellation: true, noiseSuppression: true, autoGainControl: true}
    }
    return {deviceId: {exact: selected}, echoCancellation: true, noiseSuppression: true, autoGainControl: true}
  }

  async function openCaptureStream(nextDeviceId: string): Promise<MediaStream> {
    const video = nextDeviceId ? {deviceId: {exact: nextDeviceId}} : {facingMode: "environment"}
    try {
      return await navigator.mediaDevices.getUserMedia({audio: audioConstraints(), video})
    } catch (error) {
      if (!audioDeviceIdRef.current) throw error
      audioDeviceIdRef.current = ""
      setAudioDeviceId("")
      return navigator.mediaDevices.getUserMedia({
        audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true},
        video,
      })
    }
  }

  async function startStream(nextDeviceId: string): Promise<MediaStream> {
    stopStream()
    let stream = await openCaptureStream(nextDeviceId)
    const activeLabel = stream.getAudioTracks()[0]?.label || ""
    if (!audioChoiceExplicitRef.current && /dock|virtual|zoom/i.test(activeLabel)) {
      const microphones = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === "audioinput")
      const picked = pickAudioInputId(microphones, "")
      setAudioDevices(microphones)
      if (picked && picked !== "default") {
        audioDeviceIdRef.current = picked
        setAudioDeviceId(picked)
        stream.getTracks().forEach((track) => track.stop())
        stream = await openCaptureStream(nextDeviceId)
      }
    }
    const activeAudioId = stream.getAudioTracks()[0]?.getSettings().deviceId
    if (activeAudioId) {
      audioDeviceIdRef.current = activeAudioId
      setAudioDeviceId(activeAudioId)
    }
    streamRef.current = stream
    if (videoRef.current) {
      videoRef.current.srcObject = stream
      await videoRef.current.play().catch(() => undefined)
    }
    const cameras = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === "videoinput")
    setDevices(cameras)
    if (!nextDeviceId) {
      const activeId = stream.getVideoTracks()[0]?.getSettings().deviceId
      if (activeId) setDeviceId(activeId)
    }
    return stream
  }

  async function ensureStream(): Promise<MediaStream> {
    if (streamRef.current) return streamRef.current
    return startStream(deviceId)
  }

  async function beginHold() {
    if (busyRef.current || recordingRef.current || armingRef.current) return
    const hold = {current: true}
    holdRef.current = hold
    armingRef.current = true
    cancelSpeech()
    setAnswerText("")
    setStatusText(hasFrameRef.current ? "Recording a follow-up question." : "Recording your question.")
    try {
      const stream = await ensureStream()
      if (!hold.current) {
        setStatusText(
          hasFrameRef.current
            ? "Ask a follow-up, or take a new photo."
            : "Press and hold to ask about what the camera sees.",
        )
        return
      }
      chunksRef.current = []
      const recorder = startRecorder(stream, (blob) => {
        chunksRef.current.push(blob)
      })
      recorderRef.current = recorder
      recordingRef.current = true
      setRecording(true)
      if (!hold.current) endHold()
    } catch (error) {
      const message = error instanceof Error ? error.message : "The camera or microphone is unavailable."
      recordingRef.current = false
      setRecording(false)
      setStatusText(message)
      void speakText(message)
    } finally {
      armingRef.current = false
    }
  }

  function endHold() {
    if (holdRef.current) holdRef.current.current = false
    const recorder = recorderRef.current
    if (!recorder || recorder.state === "inactive") {
      recordingRef.current = false
      setRecording(false)
      return
    }
    recorder.onstop = () => {
      recordingRef.current = false
      setRecording(false)
      const blob = new Blob(chunksRef.current, {type: recorder.mimeType || "audio/webm"})
      chunksRef.current = []
      void submitQuestion(blob)
    }
    recorder.stop()
  }

  async function submitQuestion(audio: Blob) {
    if (audio.size === 0) {
      setStatusText("I could not hear a question. Press and hold, then speak.")
      return
    }

    busyRef.current = true
    setBusy(true)
    const followUp = hasFrameRef.current
    setStatusText("Transcribing your question.")
    let speechStartedAt: number | null = null
    let question = ""
    let answer = ""
    let timings: AskTimings | null = null

    try {
      const form = new FormData()
      form.set("sessionId", sessionIdRef.current)
      form.set("followUp", followUp ? "true" : "false")
      form.set("audio", audio, "question.webm")
      if (!followUp) {
        const video = videoRef.current
        if (video && video.videoWidth === 0) {
          await new Promise((resolve) => setTimeout(resolve, 400))
        }
        const frame = await resizeVideoFrame(videoRef.current as HTMLVideoElement)
        captureAtRef.current = Date.now()
        form.set("captureAt", String(captureAtRef.current))
        form.set("frame", frame, "frame.jpg")
      } else if (captureAtRef.current) {
        form.set("captureAt", String(captureAtRef.current))
      }

      const response = await fetch("/api/vocalens/ask", {method: "POST", body: form})
      const contentType = response.headers.get("content-type") || ""
      if (!response.ok || contentType.includes("application/json")) {
        const payload = (await response.json().catch(() => ({}))) as {error?: string}
        throw new Error(payload.error || "The question could not be sent.")
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error("The answer stream is unavailable.")
      const decoder = new TextDecoder()
      let buffer = ""
      const handleLine = (line: string) => {
        if (!line.trim()) return
        const event = JSON.parse(line) as AskEvent
        if (event.type === "frame_stored" && !followUp) {
          hasFrameRef.current = true
          setHasFrame(true)
        }
        if (event.type === "transcript" && event.text) {
          question = event.text
          setStatusText("Looking at the photo.")
        }
        if (event.type === "sentence" && event.text) {
          if (speechStartedAt === null) speechStartedAt = Date.now()
          answer = answer ? `${answer} ${event.text}` : event.text
          setAnswerText(answer)
          setStatusText("Speaking the answer.")
          void speakText(event.text)
        }
        if (event.type === "done") {
          question = event.question || question
          answer = event.answer || answer
          timings = event.timings || null
          setAnswerText(answer)
          setStatusText("Ask a follow-up, or take a new photo.")
        }
        if (event.type === "error") {
          throw new Error(event.message || "The question could not be answered.")
        }
      }

      while (true) {
        const {done, value} = await reader.read()
        if (done) break
        buffer += decoder.decode(value, {stream: true})
        const lines = buffer.split("\n")
        buffer = lines.pop() ?? ""
        for (const line of lines) handleLine(line)
      }
      if (buffer.trim()) handleLine(buffer)

      if (timings && question && answer) {
        const timingFields = timings as AskTimings
        await fetch("/api/vocalens/timings", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            sessionId: sessionIdRef.current,
            question,
            answer,
            ...timingFields,
            speechStartedAt,
          }),
        }).catch(() => undefined)
      }
    } catch (error) {
      let message = error instanceof Error ? error.message : "The question could not be answered."
      if (message.includes("microphone recording was silent")) {
        const label = streamRef.current?.getAudioTracks()[0]?.label
        if (label) message = `${message} Recording from ${label}.`
      }
      cancelSpeech()
      setStatusText(message)
      setAnswerText(message)
      void speakText(message)
    } finally {
      busyRef.current = false
      setBusy(false)
    }
  }

  async function startNewPhoto() {
    cancelSpeech()
    hasFrameRef.current = false
    captureAtRef.current = null
    setHasFrame(false)
    setAnswerText("")
    setStatusText("The next hold will take a new photo.")
    await fetch("/api/vocalens/session/clear", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({sessionId: sessionIdRef.current}),
    }).catch(() => undefined)
  }

  async function onMicrophoneChange(nextDeviceId: string) {
    audioChoiceExplicitRef.current = true
    audioDeviceIdRef.current = nextDeviceId
    setAudioDeviceId(nextDeviceId)
    if (!streamRef.current || recordingRef.current || busyRef.current) return
    try {
      await startStream(deviceId)
    } catch (error) {
      const message = error instanceof Error ? error.message : "That microphone could not be opened."
      setStatusText(message)
    }
  }

  async function onCameraChange(nextDeviceId: string) {
    setDeviceId(nextDeviceId)
    if (!streamRef.current || recordingRef.current || busyRef.current) return
    try {
      await startStream(nextDeviceId)
    } catch (error) {
      const message = error instanceof Error ? error.message : "That camera could not be opened."
      setStatusText(message)
    }
  }

  const holdLabel = hasFrame
    ? "Press and hold to ask a follow-up about the same photo"
    : "Press and hold to ask about what the camera sees"

  return (
    <section
      aria-labelledby="vocalens-heading"
      style={{
        background: "#ffffff",
        border: "4px solid #000000",
        padding: "0.75rem",
        marginBottom: "0.75rem",
        boxShadow: "6px 6px 0 #FF6600",
      }}
    >
      <h2
        id="vocalens-heading"
        style={{
          fontSize: "0.85rem",
          color: "#000000",
          letterSpacing: "0.08em",
          fontWeight: 900,
          textTransform: "uppercase",
          margin: "0 0 0.65rem 0",
        }}
      >
        {hasFrame ? "Follow-up question" : "Ask about this photo"}
      </h2>

      <video
        ref={videoRef}
        aria-hidden="true"
        muted
        autoPlay
        playsInline
        style={{
          width: "100%",
          maxHeight: 240,
          background: "#000000",
          border: "4px solid #000000",
          marginBottom: "0.75rem",
          objectFit: "cover",
        }}
      />

      <label htmlFor="vocalens-microphone" style={fieldLabelStyle}>
        Microphone
      </label>
      <select
        id="vocalens-microphone"
        value={audioDeviceId}
        disabled={recording || busy}
        onChange={(event) => {
          void onMicrophoneChange(event.target.value)
        }}
        style={{...controlStyle, marginBottom: "0.75rem"}}
      >
        {audioDevices.length === 0 ? <option value="">Computer microphone</option> : null}
        {audioDevices.map((device, index) => (
          <option key={device.deviceId || index} value={device.deviceId}>
            {device.label || `Microphone ${index + 1}`}
          </option>
        ))}
      </select>

      <label htmlFor="vocalens-camera" style={fieldLabelStyle}>
        Camera
      </label>
      <select
        id="vocalens-camera"
        value={deviceId}
        disabled={recording || busy}
        onChange={(event) => {
          void onCameraChange(event.target.value)
        }}
        style={{...controlStyle, marginBottom: "0.75rem"}}
      >
        {devices.length === 0 ? <option value="">Phone camera</option> : null}
        {devices.map((device, index) => (
          <option key={device.deviceId || index} value={device.deviceId}>
            {device.label || `Camera ${index + 1}`}
          </option>
        ))}
      </select>

      <button
        type="button"
        aria-label={holdLabel}
        aria-pressed={recording}
        disabled={busy}
        onPointerDown={(event) => {
          event.preventDefault()
          event.currentTarget.setPointerCapture(event.pointerId)
          void beginHold()
        }}
        onPointerUp={(event) => {
          event.preventDefault()
          endHold()
        }}
        onPointerCancel={() => endHold()}
        onKeyDown={(event) => {
          if (event.key !== " " && event.key !== "Enter") return
          if (event.repeat) return
          event.preventDefault()
          void beginHold()
        }}
        onKeyUp={(event) => {
          if (event.key !== " " && event.key !== "Enter") return
          event.preventDefault()
          endHold()
        }}
        style={{
          width: "100%",
          minHeight: "4.5rem",
          background: recording ? "#000000" : "#FF6600",
          color: recording ? "#ffffff" : "#000000",
          border: "4px solid #000000",
          fontFamily: "inherit",
          fontWeight: 900,
          fontSize: "1rem",
          letterSpacing: "0.04em",
          boxShadow: "6px 6px 0 #000000",
          touchAction: "none",
          cursor: busy ? "wait" : "pointer",
          marginBottom: "0.75rem",
        }}
      >
        {recording ? "Release to send" : busy ? "Working on the answer" : hasFrame ? "Hold to ask a follow-up" : "Hold to ask"}
      </button>

      <button
        type="button"
        aria-label="Take a new photo and start a new question"
        disabled={!hasFrame || busy || recording}
        onClick={() => {
          void startNewPhoto()
        }}
        style={{
          ...controlStyle,
          cursor: !hasFrame || busy || recording ? "not-allowed" : "pointer",
          opacity: !hasFrame || busy || recording ? 0.55 : 1,
          marginBottom: "0.75rem",
        }}
      >
        New photo
      </button>

      <label htmlFor="vocalens-speech-rate" style={fieldLabelStyle}>
        Speech rate, {rate.toFixed(2)} times
      </label>
      <input
        id="vocalens-speech-rate"
        type="range"
        min={MIN_SPEECH_RATE}
        max={MAX_SPEECH_RATE}
        step={0.05}
        value={rate}
        aria-valuemin={MIN_SPEECH_RATE}
        aria-valuemax={MAX_SPEECH_RATE}
        aria-valuenow={rate}
        aria-valuetext={`${rate.toFixed(2)} times`}
        onChange={(event) => {
          const next = writeSpeechRate(Number(event.target.value))
          setRate(next)
        }}
        style={{width: "100%", accentColor: "#000000", marginBottom: "0.75rem"}}
      />

      <p aria-live="polite" role="status" style={{fontSize: "0.85rem", color: "#000000", fontWeight: 700, margin: "0 0 0.45rem 0"}}>
        {statusText}
      </p>
      <p aria-live="assertive" style={{fontSize: "0.95rem", color: "#000000", fontWeight: 900, margin: 0}}>
        {answerText}
      </p>
    </section>
  )
}
