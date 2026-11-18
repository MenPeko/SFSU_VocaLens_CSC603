/**
 * VocaLens — one local Bun server.
 *
 * Port 3333 serves the React webview, the API routes (including SSE), and
 * the MediaPipe files under /mediapipe/*. The laptop camera is the only
 * sign-capture source. Speech plays in the browser.
 */

import {serve} from "bun"
import {routes} from "./api/routes"
import index from "./webview/index.html"

const PORT = parseInt(process.env.PORT || "3333", 10)

console.log("Starting VocaLens...\n")

async function serveMediapipeAsset(req: Request): Promise<Response> {
  const pathname = new URL(req.url).pathname
  if (!pathname.startsWith("/mediapipe/") || pathname.includes("..") || pathname.includes("\\")) {
    return new Response("Not found", {status: 404})
  }

  const file = Bun.file(`./public${pathname}`)
  if (!(await file.exists())) {
    return new Response("Not found", {status: 404})
  }

  return new Response(file)
}

const server = serve({
  port: PORT,
  routes: {
    ...routes,

    // MediaPipe loads these scripts when the camera starts. They must be the
    // files in public/, not the HTML app shell from the catch-all below.
    "/mediapipe/*": {
      GET: serveMediapipeAsset,
    },

    // Serve webview as fallback (Bun handles JSX/Tailwind automatically)
    "/*": index,
  },

  development: process.env.NODE_ENV !== "production" && {
    hmr: true,
    console: true,
  },
})

console.log(`VocaLens is ready at ${server.url}`)
console.log(`   - Webview: ${server.url}`)
console.log(`   - API: ${server.url}api/hello`)
console.log(`   - Stream: ${server.url}api/stream\n`)

const shutdown = () => {
  console.log("\nShutting down...")
  server.stop()
  process.exit(0)
}

process.on("SIGINT", shutdown)
process.on("SIGTERM", shutdown)
