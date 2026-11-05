Place MediaPipe runtime files in this directory so the browser can fetch them.

For the ASL preview hook, the expected path is:

- `public/mediapipe/hands/*`
- `public/mediapipe/holistic/*`

The hook resolves assets such as:

- `/mediapipe/hands/hands.binarypb`
- `/mediapipe/hands/hands_solution_wasm_bin.wasm`
- `/mediapipe/hands/hands_solution_simd_wasm_bin.wasm`
- `/mediapipe/holistic/holistic.binarypb`
- `/mediapipe/holistic/holistic_solution_wasm_bin.wasm`
- `/mediapipe/holistic/holistic_solution_simd_wasm_bin.wasm`

During local setup, copy the contents of:

- `node_modules/@mediapipe/hands/` into `public/mediapipe/hands/`
- `node_modules/@mediapipe/holistic/` into `public/mediapipe/holistic/`
