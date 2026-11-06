"""
Web UI for the optical-flow frame interpolation project.

Run (from this folder, with the virtual environment active):
    pip install streamlit
    streamlit run app.py
"""
import os
import tempfile
from pathlib import Path

import streamlit as st

from ui_logic import (brightness_demo, compare_at, load_frames, make_slowmo, motion_scores, write_h264)

HERE = Path(__file__).parent


def read_text_safe(path):
    """Read a text file written by evaluate.py on Windows (may not be UTF-8)."""
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")

st.set_page_config(page_title="Video interpolation with optical flow", layout="wide")
st.title("Video frame interpolation with optical flow")
st.caption("Hide a real frame, rebuild it from its two neighbours, and compare the simple cross-fade "
           "with optical flow (DIS / Farneback).")


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Video")
    source = st.radio("Where is the video?", ["Upload a file", "Browse the UCF-101 folder"])
    video_path = None
    if source == "Upload a file":
        up = st.file_uploader("Video file", type=["avi", "mp4", "mov", "mkv"])
        if up is not None:
            suffix = os.path.splitext(up.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(up.getbuffer())
            tmp.close()
            video_path = tmp.name
    else:
        root = st.text_input("UCF-101 folder", value=str(HERE / "UCF-101"))
        if os.path.isdir(root):
            classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
            if classes:
                cls = st.selectbox("Action class", classes)
                vids = sorted(f for f in os.listdir(os.path.join(root, cls))
                              if f.lower().endswith((".avi", ".mp4")))
                if vids:
                    vid = st.selectbox("Video", vids)
                    video_path = os.path.join(root, cls, vid)
        else:
            st.warning("Folder not found. Type the full path to your UCF-101 folder.")

    st.header("2. Settings")
    gap = st.selectbox("Gap (frames between A and the hidden frame)", [1, 2, 3], index=1,
                       help="1 = neighbouring frames. 2 and 3 = more motion, a harder test.")
    method = st.radio("Optical-flow method", ["dis", "farneback", "fused"], horizontal=True)
    alpha = st.slider("Occlusion strength (alpha)", 0.0, 2.0, 0.5, 0.1,
                      help="How strongly pixels with unreliable flow are down-weighted. The project used 0.5.")
    max_frames = st.slider("Max frames to load", 30, 300, 150, 10)
    max_width = st.select_slider("Shrink wide videos to (px)", options=[320, 480, 640, 960, 1280], value=640,
                                 help="UCF-101 clips are 320 px wide, so they are not changed.")


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Reading video...")
def _load(path, max_frames, max_width):
    return load_frames(path, max_frames, max_width)


@st.cache_data(show_spinner="Looking for the highest-motion moment...")
def _motion(path, max_frames, max_width, gap):
    frames, _ = _load(path, max_frames, max_width)
    return motion_scores(frames, gap)


@st.cache_data(show_spinner="Interpolating...")
def _compare(path, max_frames, max_width, start, gap, method, alpha):
    frames, _ = _load(path, max_frames, max_width)
    return compare_at(frames, start, gap, method, alpha)


tab_cmp, tab_flow, tab_slow, tab_res, tab_extra = st.tabs(
    ["Compare methods", "Optical-flow arrows", "Slow motion", "Saved results", "Extra experiments"])

frames, fps, n_start = None, 25.0, 0
if video_path:
    frames, fps = _load(video_path, max_frames, max_width)
    n_start = len(frames) - (2 * gap + 1)
    if n_start < 0:
        st.error("This video is too short for the chosen gap.")
        frames = None

# ----------------------------------------------------------------------------- tab 1: compare
with tab_cmp:
    if frames is None:
        st.info("Choose a video in the sidebar to start.")
    else:
        if "start" not in st.session_state or st.session_state.start > n_start:
            st.session_state.start = 0

        def _jump():
            starts, motions = _motion(video_path, max_frames, max_width, gap)
            st.session_state.start = int(starts[motions.index(max(motions))])

        c1, c2 = st.columns([3, 1])
        c1.slider("Start frame (A)", 0, max(n_start, 1), key="start")
        c2.write("")
        c2.button("Jump to highest-motion moment", on_click=_jump)

        s = st.session_state.start
        r = _compare(video_path, max_frames, max_width, s, gap, method, alpha)
        st.write(f"Frame A = #{s}, hidden real frame = #{s + gap}, frame B = #{s + 2 * gap}. "
                 f"Average motion between A and B: **{r['motion_px']:.1f} px**.")

        st.subheader("The real frames")
        a, g, b = st.columns(3)
        a.image(r["A"], caption="Frame A", channels="BGR")
        g.image(r["GT"], caption="Ground truth (the real middle frame, hidden from the program)", channels="BGR")
        b.image(r["B"], caption="Frame B", channels="BGR")

        st.subheader("What the methods rebuilt")
        c, f, e = st.columns(3)
        c.image(r["crossfade"], caption=f"Cross-fade  |  PSNR {r['crossfade_psnr']:.2f} dB  |  "
                                        f"SSIM {r['crossfade_ssim']:.4f}", channels="BGR")
        f.image(r["flow"], caption=f"Optical flow ({method})  |  PSNR {r['flow_psnr']:.2f} dB  |  "
                                   f"SSIM {r['flow_ssim']:.4f}", channels="BGR")
        e.image(r["error"], caption="Error of optical flow x4 (dark = correct, bright = mistake)", channels="BGR")

        d = r["flow_psnr"] - r["crossfade_psnr"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Cross-fade PSNR", f"{r['crossfade_psnr']:.2f} dB")
        m2.metric("Optical-flow PSNR", f"{r['flow_psnr']:.2f} dB", f"{d:+.2f} dB vs cross-fade")
        m3.metric("One-way warp PSNR", f"{r['oneway_psnr']:.2f} dB")
        st.caption("This is one moment of one video. The averages over 600 test triplets are in the "
                   "'Saved results' tab, and those are the numbers to quote.")

        with st.expander("What do these words mean?"):
            st.markdown(
                "- **Ground truth**: the real frame that was in the video. The program never sees it; it is only "
                "used to check the answer.\n"
                "- **Cross-fade**: the simple baseline, a 50/50 blend of A and B. Moving objects look doubled (ghosting).\n"
                "- **Optical flow**: an arrow for every pixel saying where it moved. We move pixels along the arrows "
                "instead of blending.\n"
                "- **PSNR (dB)**: how close the rebuilt frame is to the real one, pixel by pixel. Higher is better.\n"
                "- **SSIM (0 to 1)**: whether edges and structures look alike. 1 means identical.\n"
                "- **One-way warp**: warp only frame A (no blending with B, no occlusion handling).\n"
                "- **Gap**: with gap g, A and B are 2g frames apart and the hidden frame is in the middle.")

# ----------------------------------------------------------------------------- tab 2: flow
with tab_flow:
    if frames is None:
        st.info("Choose a video in the sidebar to start.")
    else:
        st.write("Colour shows the direction each pixel moves, and brightness shows how far. "
                 "Black means no movement. This is the forward flow from frame A to frame B at the moment "
                 "chosen in the first tab.")
        r = _compare(video_path, max_frames, max_width, st.session_state.start, gap, method, alpha)
        x, y = st.columns(2)
        x.image(r["A"], caption="Frame A", channels="BGR")
        y.image(r["flow_color"], caption=f"Optical flow {method}: A to B", channels="BGR")

# ----------------------------------------------------------------------------- tab 3: slow motion
with tab_slow:
    if frames is None:
        st.info("Choose a video in the sidebar to start.")
    else:
        st.write("Insert new frames between every pair of real frames and play the result at the original "
                 "frame rate, which makes it slow motion.")
        factor = st.selectbox("Slow down by", [2, 3, 4], index=1)
        lo, hi = st.slider("Which frames to use (keep it short, it takes a while)", 0, len(frames) - 1,
                           (0, min(len(frames) - 1, 30)))
        if st.button("Make slow-motion video"):
            clip = frames[lo:hi + 1]
            if len(clip) < 2:
                st.error("Pick at least two frames.")
            else:
                bar = st.progress(0.0, text="Interpolating...")
                new = make_slowmo(clip, factor, method, alpha, progress=lambda p: bar.progress(p))
                bar.empty()
                out_path = os.path.join(tempfile.gettempdir(), "slowmo_ui.mp4")
                if write_h264(new, fps, out_path):
                    with open(out_path, "rb") as fh:
                        data = fh.read()
                    st.video(data)
                    st.download_button("Download the video", data, file_name="slow_motion.mp4",
                                       mime="video/mp4")
                else:
                    st.error("Could not write the video. Install the encoder with:  pip install imageio-ffmpeg")

# ----------------------------------------------------------------------------- tab 4: results
with tab_res:
    st.write("Results of the full evaluation on UCF-101 (200 random videos x 3 triplets = 600 triplets per gap).")
    found = False
    for g in (1, 2, 3):
        d = HERE / f"results_gap{g}"
        md = d / "summary.md"
        if md.exists():
            found = True
            with st.expander(f"Gap {g}  (A and B are {2 * g} frames apart)", expanded=(g == 1)):
                st.markdown(read_text_safe(md))
                cols = st.columns(3)
                for col, name in zip(cols, ["metrics_bar.png", "psnr_gain_hist.png", "gain_vs_motion.png"]):
                    if (d / name).exists():
                        col.image(str(d / name), caption=name)
    rt = HERE / "results_runtime" / "runtime_table.md"
    if rt.exists():
        found = True
        with st.expander("Running time"):
            st.markdown(read_text_safe(rt))
            if (rt.parent / "runtime_loglog.png").exists():
                st.image(str(rt.parent / "runtime_loglog.png"))
    if not found:
        st.info("No result folders found next to app.py. Run evaluate.py first (see README).")


# ----------------------------------------------------------------------------- tab 5: extra experiments
with tab_extra:
    st.subheader("Live demo: what happens if the brightness changes between the frames?")
    st.write("Optical flow assumes a pixel keeps its brightness while it moves. Here the brightness drifts "
             "from frame A (+0) through the hidden frame (+d/2) to frame B (+d), so you can see how flow and the "
             "cross-fade react. Uses the moment chosen in the first tab.")
    if frames is None:
        st.info("Choose a video in the sidebar to try the live demo.")
    else:
        delta = st.slider("Brightness change d (grey levels)", 0, 80, 40, 5)
        bd = brightness_demo(frames, st.session_state.start, gap, method, alpha, delta)
        p1, p2, p3 = st.columns(3)
        p1.image(bd["GT"], caption="Ground truth (brightness shifted by d/2)", channels="BGR")
        p2.image(bd["crossfade"], caption=f"Cross-fade  |  PSNR {bd['crossfade_psnr']:.2f} dB", channels="BGR")
        p3.image(bd["flow"], caption=f"Optical flow ({method})  |  PSNR {bd['flow_psnr']:.2f} dB", channels="BGR")
        st.caption("One moment of one video, so it is an illustration. The averaged result over many test "
                   "triplets is in the saved table below.")

    st.subheader("Saved results of the extra experiments")
    st.write("These take several minutes each, so they are run once with `python extras.py ...` and shown here. "
             "See the README for the commands.")
    titles = {"ablation": "Ablation: what does each part of the method add?",
              "alpha": "Choosing the occlusion strength on a separate validation set",
              "hybrid": "Motion-aware hybrid (cross-fade for still scenes)",
              "brightness": "Robustness to a brightness change"}
    any_extra = False
    for g in (1, 2, 3):
        d = HERE / f"extras_gap{g}"
        for key, title in titles.items():
            f = d / f"{key}_gap{g}.md"
            if f.exists():
                any_extra = True
                with st.expander(f"{title}  (gap {g})"):
                    st.markdown(read_text_safe(f))
                    img = d / f"{key}_gap{g}.png"
                    if img.exists():
                        st.image(str(img))
    if not any_extra:
        st.info("No extra results yet. Run, for example:  python extras.py ablation --data_dir UCF-101 --gap 2")        