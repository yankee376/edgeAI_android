"""Pi camera server exposing both MJPEG and a single latest JPEG snapshot.

This is a separate replacement for the existing camera_web.py. It prefers
Picamera2 for a CSI camera and falls back to OpenCV/V4L2 for a USB camera. It
keeps /video and /stats, and adds /snapshot.jpg for Android polling.
"""

import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify


CAMERA_ID = int(os.environ.get("CAMERA_ID", "0"))
WIDTH = int(os.environ.get("CAMERA_WIDTH", "640"))
HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "480"))
TARGET_FPS = int(os.environ.get("CAMERA_FPS", "25"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "75"))
PORT = int(os.environ.get("PORT", "5000"))

app = Flask(__name__)


class CameraStream:
    """Capture once and share the newest encoded frame across HTTP clients."""

    def __init__(self):
        self._capture = None
        self._picam2 = None
        self.backend = ""
        self.width = WIDTH
        self.height = HEIGHT
        self.camera_fps = float(TARGET_FPS)

        csi_error = None
        try:
            from picamera2 import Picamera2

            picam2 = Picamera2()
            config = picam2.create_video_configuration(
                main={"size": (WIDTH, HEIGHT), "format": "RGB888"},
                controls={"FrameRate": TARGET_FPS},
            )
            picam2.configure(config)
            picam2.start()
            self._picam2 = picam2
            self.backend = "Picamera2/CSI"
        except Exception as error:
            csi_error = error
            if self._picam2 is not None:
                try:
                    self._picam2.stop()
                except Exception:
                    pass
                self._picam2 = None

        if self._picam2 is None:
            capture = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
            capture.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG"),
            )
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            capture.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(
                    "Không mở được camera CSI bằng Picamera2 hoặc camera "
                    f"V4L2 id={CAMERA_ID}. Lỗi Picamera2: {csi_error}"
                )
            self._capture = capture
            self.backend = "OpenCV/V4L2"
            self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.camera_fps = capture.get(cv2.CAP_PROP_FPS)
        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._frame_id = 0
        self._captured_monotonic_ns = 0
        self._fps_actual = 0.0
        self._running = False
        self._thread = None
        self._fps_frames = 0
        self._fps_started = time.perf_counter()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._capture is not None:
            self._capture.release()
        if self._picam2 is not None:
            self._picam2.stop()

    def _capture_loop(self):
        while self._running:
            if self._picam2 is not None:
                try:
                    frame = self._picam2.capture_array("main")
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    ok = True
                except Exception:
                    ok, frame = False, None
            else:
                ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.02)
                continue

            frame = cv2.rotate(frame, cv2.ROTATE_180)
            encoded, jpeg = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            if not encoded:
                continue

            captured_ns = time.monotonic_ns()
            with self._lock:
                self._latest_jpeg = jpeg.tobytes()
                self._frame_id += 1
                self._captured_monotonic_ns = captured_ns

            self._fps_frames += 1
            elapsed = time.perf_counter() - self._fps_started
            if elapsed >= 1.0:
                with self._lock:
                    self._fps_actual = self._fps_frames / elapsed
                self._fps_frames = 0
                self._fps_started = time.perf_counter()

    def latest(self):
        with self._lock:
            return (
                self._latest_jpeg,
                self._frame_id,
                self._captured_monotonic_ns,
            )

    def stats(self):
        with self._lock:
            return {
                "actual_fps": round(self._fps_actual, 2),
                "width": self.width,
                "height": self.height,
                "camera_fps": self.camera_fps,
                "frame_id": self._frame_id,
                "backend": self.backend,
            }


camera = CameraStream()


@app.get("/")
def index():
    return (
        "Pi camera server is running. "
        "Use /snapshot.jpg for one JPEG, /video for MJPEG, /stats for JSON."
    )


@app.get("/snapshot.jpg")
def snapshot():
    jpeg, frame_id, captured_ns = camera.latest()
    if jpeg is None:
        return "Camera chưa có frame", 503
    return Response(
        jpeg,
        mimetype="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "X-Frame-Id": str(frame_id),
            "X-Captured-Monotonic-Ns": str(captured_ns),
        },
    )


@app.get("/video")
def video():
    def generate():
        last_frame_id = -1
        while True:
            jpeg, frame_id, _ = camera.latest()
            if jpeg is None or frame_id == last_frame_id:
                time.sleep(0.01)
                continue
            last_frame_id = frame_id
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stats")
def stats():
    return jsonify(camera.stats())


if __name__ == "__main__":
    print(
        f"Camera ready: {camera.width}x{camera.height}; "
        f"requested FPS={TARGET_FPS}; HTTP port={PORT}"
    )
    camera.start()
    try:
        app.run(
            host="0.0.0.0",
            port=PORT,
            threaded=True,
            debug=False,
            use_reloader=False,
        )
    finally:
        camera.stop()
