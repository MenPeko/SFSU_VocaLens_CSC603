# How It Works: Bun 1.3 React App Serving

This document explains how VocaLens uses Bun 1.3's built-in features to serve a React application with **zero build step**.

The running app is one Bun process, started from `src/index.ts` on `PORT` (default `3333`). That process serves the page, the API, and `/mediapipe/*`. Live Sign reads the laptop camera in the page. Speech uses the browser's `speechSynthesis` through `speakText`.

## Overview

Traditional React apps require:
- Build tools (Webpack, Vite, etc.)
- Transpilation step (Babel)
- CSS processing (PostCSS)
- Bundle output to disk

**Bun 1.3 eliminates all of this.** It serves, transpiles, and processes everything on-the-fly.

## The Magic: Three Files

### 1. `bunfig.toml` - Configuration

```toml
[serve.static]
plugins = ["bun-plugin-tailwind"]
env = "BUN_PUBLIC_*"
```

**What it does:**
- Tells Bun to use the Tailwind plugin when serving static files
- Processes CSS with Tailwind directives (`@apply`, `@layer`, etc.)
- Enables environment variable prefixes

**Without this file:** Tailwind won't work! CSS files will be served raw without processing.

### 2. `src/index.ts` - Server Setup

```typescript
import {serve} from "bun"
import {routes} from "./api/routes"
import index from "./webview/index.html"

const PORT = parseInt(process.env.PORT || "3333", 10)

const server = serve({
  port: PORT,
  routes: {
    ...routes,
    "/mediapipe/*": {
      GET: serveMediapipeAsset,
    },
    "/*": index,
  },
  development: process.env.NODE_ENV !== "production" && {
    hmr: true,
    console: true,
  },
})
```

There is no second server. `routes` already includes `/api/stream`, so translation events reach the page from this same process. `serveMediapipeAsset`, defined in that file, returns the matching file from `public/`. The laptop camera and browser speech live in the webview, not in this file.

**What `import index from "./webview/index.html"` does:**
- Bun reads the HTML file
- Finds all `<script>` and `<link>` tags
- Resolves relative paths (like `./frontend.tsx`)
- Sets up automatic transpilation for those files
- Configures HMR for hot reloading

### 3. `src/webview/index.html` - Entry Point

```html
<!doctype html>
<html lang="en">
  <head>
    <script type="module" src="./frontend.tsx" async></script>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
```

**What happens:**
1. Browser loads HTML
2. Browser requests `./frontend.tsx`
3. Bun intercepts the request
4. Bun transpiles TSX → JS on-the-fly
5. Browser receives JavaScript
6. React app starts!

## The Flow

### Development Mode (`bun --hot src/index.ts`)

```
1. Server starts
   └─> Bun reads bunfig.toml
       └─> Loads Tailwind plugin

2. Browser requests "/"
   └─> Server returns index.html

3. Browser requests "./frontend.tsx"
   └─> Bun transpiles TSX to JS
   └─> Bun injects HMR code
   └─> Returns JavaScript

4. Browser requests "./index.css"
   └─> Bun processes Tailwind directives
   └─> Returns compiled CSS

5. You edit App.tsx
   └─> HMR detects change
   └─> Browser hot-reloads (no refresh!)
```

### File Resolution

When HTML references `./frontend.tsx`, Bun looks for it **relative to the HTML file's location**:

```
src/webview/index.html
     └─> references "./frontend.tsx"
         └─> Bun resolves to: src/webview/frontend.tsx ✓
```

**This is why everything must be in the same directory!**

If you import from a different location, the relative paths break:

```typescript
// ❌ WRONG - Breaks relative paths
import index from "../other/index.html"

// ✓ CORRECT - Same directory
import index from "./webview/index.html"
```

## Bun's Built-in Features

### 1. JSX/TSX Transpilation

Bun automatically transpiles:
- `.jsx` → JavaScript
- `.tsx` → JavaScript
- TypeScript → JavaScript

**No Babel needed!** It's built into Bun.

### 2. CSS Processing

With `bun-plugin-tailwind` in `bunfig.toml`:
- Processes `@import` statements
- Handles Tailwind directives (`@apply`, `@layer`)
- Minifies CSS in production
- Adds vendor prefixes

### 3. Hot Module Reloading (HMR)

```typescript
development: {
  hmr: true,
  console: true,
}
```

**How it works:**
1. Bun watches your source files
2. When you save a change, Bun detects it
3. Bun sends a message to the browser via WebSocket
4. Browser hot-swaps the module (no page refresh!)

React components use this pattern:

```typescript
if (import.meta.hot) {
  const root = (import.meta.hot.data.root ??= createRoot(elem));
  root.render(app);
}
```

This preserves the React root instance across HMR updates.

### 4. Routes API

Bun 1.2+ introduced the `routes` object:

```typescript
serve({
  routes: {
    "/api/hello": {
      GET: () => Response.json({ msg: "hi" }),
      POST: () => Response.json({ msg: "posted" }),
    },
    "/users/:id": (req) => {
      return Response.json({ id: req.params.id });
    },
    "/*": index,  // Fallback (must be last!)
  }
});
```

**Features:**
- Pattern matching (`/users/:id`)
- HTTP method handlers (`GET`, `POST`, etc.)
- Wildcard matching (`/*`)
- Order matters! Routes are checked top-to-bottom

## Common Issues & Solutions

### Issue: Tailwind styles not loading

**Symptom:** Unstyled content, raw HTML
**Cause:** Missing `bunfig.toml`
**Solution:** Add `bunfig.toml` with Tailwind plugin

```toml
[serve.static]
plugins = ["bun-plugin-tailwind"]
```

### Issue: "Cannot GET /"

**Symptom:** 404 error on homepage
**Cause:** Wildcard route `"/*"` not configured or placed before API routes
**Solution:** Put `"/*": index` **last** in routes object

```typescript
routes: {
  "/api/*": apiRoutes,  // Specific routes first
  "/*": index,          // Wildcard last!
}
```

### Issue: Module not found

**Symptom:** `Cannot find module './Component.tsx'`
**Cause:** Relative path broken due to HTML import location
**Solution:** Move HTML to same directory as server file

### Issue: HMR not working

**Symptom:** Changes don't reload automatically
**Cause:** Missing `development` config or wrong `NODE_ENV`
**Solution:** Ensure `NODE_ENV !== "production"` and:

```typescript
development: process.env.NODE_ENV !== "production" && {
  hmr: true,
  console: true,
}
```

## Comparison: Traditional vs Bun

### Traditional React App

```
1. Write code
2. Run build (webpack/vite)
   - Transpile TS
   - Bundle modules
   - Process CSS
   - Write to disk
3. Serve from dist/
4. Repeat for every change
```

**Time:** 5-30 seconds per change

### Bun 1.3 React App

```
1. Write code
2. Save
3. HMR updates instantly
```

**Time:** < 1 second

## Why This Structure Works

```
src/
├── index.ts           # One Bun server. Imports the webview HTML.
└── webview/
    ├── index.html     # References ./frontend.tsx (same dir)
    ├── frontend.tsx
    └── App.tsx        # Laptop camera and browser speech
```

**Key principles:**
1. `bunfig.toml` in project root
2. Server imports HTML
3. HTML references files with relative paths (`./`)
4. All webview files in same directory tree

## Production Deployment

For production, you can:

1. **Run directly with Bun:**
   ```bash
   NODE_ENV=production bun src/index.ts
   ```
   Bun still transpiles on-the-fly (very fast!)

2. **Pre-build with `bun build`:**
   ```bash
   bun build src/index.ts --target=bun --outdir=dist
   ```
   This pre-compiles everything for maximum performance

## Advanced: How Bun Processes HTML Imports

When you `import index from "./index.html"`:

1. **Parse:** Bun parses the HTML
2. **Extract:** Finds all `<script>` and `<link>` tags
3. **Register:** Registers those paths for transpilation
4. **Manifest:** Creates an internal manifest mapping URLs to files
5. **Serve:** When browser requests those URLs, Bun serves from manifest

This is why:
- No build step needed
- HMR works automatically
- Relative paths resolve correctly

## Resources

- [Bun Server Documentation](https://bun.sh/docs/api/http)
- [Bun HTML Imports](https://bun.sh/docs/bundler/fullstack)
- [bun-plugin-tailwind](https://www.npmjs.com/package/bun-plugin-tailwind)

## Summary

**Three ingredients make this work:**

1. ✅ `bunfig.toml` - Enables Tailwind plugin
2. ✅ `import index from "./webview/index.html"` - HTML import
3. ✅ `routes: { "/*": index }` - Route configuration

Everything else is handled by Bun automatically! 🚀