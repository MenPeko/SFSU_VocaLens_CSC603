"""TouchDesigner-oriented vision runtime.

Standalone Python process that runs MediaPipe Holistic, extracts the same
landmark subset + features as the original browser/TypeScript pipeline, and
streams derived real-time signals to TouchDesigner over OSC.
"""
