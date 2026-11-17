/**
 * Landmark extractor version lock.
 *
 * Every captured sample, every trained model artifact, and every live
 * classification request is stamped with this identifier so that downstream
 * code can detect silent drift caused by a MediaPipe version bump or a change
 * to the landmark subset we emit from the webview hook.
 *
 * The identifier has two parts:
 *
 *   mediapipe-holistic-js@<package-version>+schema-v<schema-revision>
 *
 * - The `@<package-version>` half must match the pinned version of
 *   `@mediapipe/holistic` in `package.json` exactly. If you bump the
 *   MediaPipe package, bump this string too and regenerate every sample +
 *   retrain every model that uses it.
 *
 * - The `+schema-v<n>` half covers *our* landmark subset shape: the specific
 *   face indices in `FACE_SUBSET_INDICES`, the hand/pose subsets emitted by
 *   `useSignLanguage.ts`, the configured holistic model complexity, and any
 *   other transformation we apply before the sample hits disk. Bump this if
 *   the layout or extraction regime changes even when MediaPipe itself did not.
 *
 * A matching constant exists in
 * `sign_classifier_service/extractor_version.py` — keep them in sync.
 *
 * See DEVELOPMENT_ROADMAP.md — Milestone 1 task 2 and
 * Critical Risks #2 for the rationale.
 */
export const MEDIAPIPE_EXTRACTOR_VERSION = "mediapipe-holistic-js@0.5.1675471629+schema-v2"

/**
 * Type alias for the extractor version field. Kept as a plain string so that
 * loading older sample JSON with a different lock still typechecks — the
 * runtime guards are what actually reject mismatches.
 */
export type ExtractorVersion = string
