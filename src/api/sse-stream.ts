export interface SSEClient {
  send: (data: unknown) => void
}

export type StreamEventType =
  | "connected"
  | "status"
  | "ocr_result"
  | "translation_result"
  | "camera_stream"
  | "error"
  | "heartbeat"

export interface StreamEvent {
  type: StreamEventType
  [key: string]: unknown
}

class SSEManager {
  private clients = new Set<SSEClient>()
  private heartbeatInterval: Timer | null = null

  constructor() {
    this.startHeartbeat()
  }

  addClient(client: SSEClient): void {
    this.clients.add(client)
  }

  removeClient(client: SSEClient): void {
    this.clients.delete(client)
  }

  broadcast(event: StreamEvent): void {
    for (const client of this.clients) {
      client.send(event)
    }
  }

  get clientCount(): number {
    return this.clients.size
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      this.broadcast({type: "heartbeat", timestamp: Date.now()})
    }, 15000)
  }

  dispose(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
    }
    this.clients.clear()
  }
}

// Singleton
export const sseManager = new SSEManager()

export const streamRoute = {
  "/api/stream": {
    async GET(req: Request) {
      const stream = new ReadableStream({
        start(controller) {
          const encoder = new TextEncoder()

          const connectMsg = `data: ${JSON.stringify({type: "connected"})}\n\n`
          controller.enqueue(encoder.encode(connectMsg))

          const client: SSEClient = {
            send: (data: unknown) => {
              try {
                const msg = `data: ${JSON.stringify(data)}\n\n`
                controller.enqueue(encoder.encode(msg))
              } catch {
                sseManager.removeClient(client)
              }
            },
          }

          sseManager.addClient(client)

          req.signal?.addEventListener("abort", () => {
            sseManager.removeClient(client)
            try {
              controller.close()
            } catch {
              // Already closed
            }
          })
        },
      })

      return new Response(stream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
          "X-Accel-Buffering": "no",
        },
      })
    },
  },
}
