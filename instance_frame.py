import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from nuscenes.nuscenes import NuScenes
from scene_vqa import SceneFrame, answer, summarize_scene

try:
    from openai import OpenAI
    client = OpenAI()
    USE_GPT = True
except ImportError:
    USE_GPT = False


# ------------------------
# Helper: Generate Visuals
# ------------------------
@st.cache_data
def generate_scene_visuals(_nusc, scene_index, output_dir="scene_output", fps=2, frame_skip=5):
    """
    Generate annotated front camera and LIDAR BEV video + preview image for a scene.
    Uses _nusc (NuScenes) to avoid Streamlit caching issues.
    """
    os.makedirs(output_dir, exist_ok=True)
    scene = _nusc.scene[scene_index]
    sample_token = scene['first_sample_token']

    front_frames, lidar_frames = [], []
    first_front_img, first_lidar_img = None, None
    frame_count = 0

    while sample_token:
        if frame_count % frame_skip == 0:
            sample = _nusc.get('sample', sample_token)

            # Front camera
            cam_front_token = sample['data']['CAM_FRONT']
            front_path = os.path.join(output_dir, f"front_{sample_token}.png")
            _nusc.render_sample_data(cam_front_token, out_path=front_path, with_anns=True, verbose=False)
            front_frames.append(imageio.imread(front_path))
            if first_front_img is None:
                first_front_img = front_path

            # LiDAR BEV
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_path = os.path.join(output_dir, f"lidar_{sample_token}.png")
            _nusc.render_sample_data(lidar_token, out_path=lidar_path, with_anns=True, verbose=False)
            lidar_frames.append(imageio.imread(lidar_path))
            if first_lidar_img is None:
                first_lidar_img = lidar_path

        frame_count += 1
        sample_token = sample['next']

    # Save videos
    front_video_path = os.path.join(output_dir, "front_camera.mp4")
    lidar_video_path = os.path.join(output_dir, "lidar_bev.mp4")
    imageio.mimsave(front_video_path, front_frames, fps=fps)
    imageio.mimsave(lidar_video_path, lidar_frames, fps=fps)

    return first_front_img, first_lidar_img, front_video_path, lidar_video_path


# ------------------------
# Quick Scene Summary
# ------------------------
def quick_scene_summary(scene_frame):
    full_desc = summarize_scene(scene_frame)
    if USE_GPT:
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=f"Summarize this in one short sentence for an autonomous driving report: {full_desc}"
            )
            return response.output_text
        except Exception:
            return "Scene summary unavailable (GPT error)."
    else:
        return full_desc[:200] + "..." if len(full_desc) > 200 else full_desc


# ------------------------
# Helper: Scene Snapshot (2D Plot)
# ------------------------
def generate_scene_snapshot(scene_frame, max_objects=5):
    """
    Generate a quick 2D plot of ego trajectory and top moving objects.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Ego vehicle trajectory
    ego_points = []
    for inst in scene_frame.instances:
        if inst.category_name.startswith("vehicle.car"):
            # Ego car is not in annotations, skipping
            continue
        ego_points.append(inst.locations)

    # Plot ego trajectory (just as reference origin)
    ax.scatter(0, 0, color="blue", s=100, label="Ego Vehicle (Origin)")

    # Plot top moving objects
    moving_instances = sorted(scene_frame.instances, key=lambda x: len(x.locations), reverse=True)[:max_objects]
    for inst in moving_instances:
        locs = np.array(inst.locations)
        ax.plot(locs[:, 0], locs[:, 1], marker="o", label=inst.category_name[:20])

    ax.axhline(0, color="gray", linestyle="--")
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("X (m, left/right)")
    ax.set_ylabel("Y (m, forward/back)")
    ax.set_title("Scene Snapshot (Ego + Top Moving Objects)")
    ax.legend()
    st.pyplot(fig)


# ------------------------
# Streamlit App
# ------------------------
st.title("NuScenes Visual Question Answering (VQA)")

DATA_ROOT = "/home/aleenatron/Frames_AV_research"
nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=False)
scene_index = st.selectbox("Select a scene index:", list(range(len(nusc.scene))))
scene_frame = SceneFrame(nusc, scene_index)

# ------------------------
# Questions
# ------------------------
st.subheader("Ask a Question")
prebuilt_questions = [
    "Is there a bus ahead?",
    "How many pedestrians are to my left?",
    "What objects are ahead?",
    "Which is closer, bus or truck?",
    "Describe the scene"
]
selected_question = st.selectbox("Choose a sample question:", prebuilt_questions)
custom_question = st.text_input("Or enter your own question:", "")
final_question = custom_question if custom_question.strip() else selected_question

if st.button("Get Answer"):
    st.write(f"**Answer:** {answer(scene_frame, final_question)}")

# ------------------------
# Quick Scene Summary
# ------------------------
if st.button("Quick Scene Summary"):
    summary = quick_scene_summary(scene_frame)
    st.success(f"**Scene Summary:** {summary}")

# ------------------------
# Visualization
# ------------------------
if st.checkbox("Show Scene Visualization"):
    with st.spinner("Generating visualization..."):
        front_img, lidar_img, front_video, lidar_video = generate_scene_visuals(_nusc=nusc, scene_index=scene_index)


    st.image(front_img, caption="Front Camera (Sample Frame)", use_container_width=True)
    st.image(lidar_img, caption="LIDAR BEV (Sample Frame)", use_container_width=True)

    if st.button("Generate Full Video"):
        st.video(front_video)
        st.video(lidar_video)

# ------------------------
# Scene Snapshot
# ------------------------
if st.checkbox("Show Scene Snapshot (Quick BEV)"):
    with st.spinner("Generating 2D snapshot..."):
        generate_scene_snapshot(scene_frame)
