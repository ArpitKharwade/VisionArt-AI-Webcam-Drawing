
import cv2
import av
import numpy as np
import streamlit as st
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="🎨VisionArt:AI Webcam Drawing", layout="wide")
st.title("VisionArt:AI Webcam Drawing")
st.markdown("""
🖌️ Hold a **dark blue object** (like a pen cap) in front of your webcam to draw.  
Use the sidebar to fine-tune HSV detection if tracking is unstable.
""")

# Sidebar controls
st.sidebar.header("🎛️ Controls")
color_choice = st.sidebar.radio("Choose Brush Color:", ("Blue", "Green", "Red", "Yellow"))
clear_canvas = st.sidebar.button("🧹 Clear Canvas")

# Brush colors
color_map = {
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
    "Red": (0, 0, 255),
    "Yellow": (0, 255, 255),
}
selected_color = color_map[color_choice]

# HSV sliders
st.sidebar.subheader("🎨 HSV Range (Dark Blue Detection)")
h_min = st.sidebar.slider("Hue Min", 0, 180, 100)
h_max = st.sidebar.slider("Hue Max", 0, 180, 130)
s_min = st.sidebar.slider("Saturation Min", 0, 255, 150)
s_max = st.sidebar.slider("Saturation Max", 0, 255, 255)
v_min = st.sidebar.slider("Value Min", 0, 255, 50)
v_max = st.sidebar.slider("Value Max", 0, 255, 255)


# ----------------------------
# Video Processor
# ----------------------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.kernel = np.ones((5, 5), np.uint8)
        self.color = selected_color
        self.canvas = None
        self.last_point = None
        self.clear_flag = False

        # Deque for smoothing (store last 5 centers)
        self.center_buffer = deque(maxlen=5)

    def update_color(self, new_color):
        self.color = new_color

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        if self.canvas is None:
            self.canvas = np.ones((h, w, 3), np.uint8) * 255

        if clear_canvas or self.clear_flag:
            self.canvas[:] = 255
            self.clear_flag = False

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(cnts) > 0:
            cnt = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            if radius > 5:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    self.center_buffer.append(center)

        # --- Smooth center using average of last few points ---
        if len(self.center_buffer) > 0:
            smoothed_x = int(np.mean([p[0] for p in self.center_buffer]))
            smoothed_y = int(np.mean([p[1] for p in self.center_buffer]))
            smooth_center = (smoothed_x, smoothed_y)
        else:
            smooth_center = None

        # --- Draw lines ---
        if smooth_center:
            cv2.circle(img, smooth_center, 8, (0, 255, 255), -1)
            if self.last_point:
                dist = np.linalg.norm(np.array(smooth_center) - np.array(self.last_point))
                if dist > 3:  # only draw if hand moved significantly
                    cv2.line(self.canvas, self.last_point, smooth_center, self.color, 6, cv2.LINE_AA)
            self.last_point = smooth_center
        else:
            self.last_point = None

        # Merge live + canvas
        combined = cv2.addWeighted(img, 0.7, self.canvas, 0.3, 0)
        return av.VideoFrame.from_ndarray(combined, format="bgr24")


# ----------------------------
# Start Streamlit WebRTC
# ----------------------------
ctx = webrtc_streamer(
    key="smooth-drawing",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.update_color(selected_color)
    if clear_canvas:
        ctx.video_processor.clear_flag = True

