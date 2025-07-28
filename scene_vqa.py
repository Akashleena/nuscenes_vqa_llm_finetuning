import numpy as np
from nuscenes.nuscenes import NuScenes
from instance_frame import compute_scene_ego_positions, InstanceFrame

# --------------------------
# SceneFrame Aggregator
# --------------------------
class SceneFrame:
    def __init__(self, nusc, scene_index):
        self.nusc = nusc
        self.scene_index = scene_index
        self.scene_ego_positions, self.ego_start_time = compute_scene_ego_positions(nusc, scene_index)
        self.instances = self._build_instances()

    def _build_instances(self):
        scene = self.nusc.scene[self.scene_index]
        sample_token = scene['first_sample_token']
        annotations = []

        while sample_token:
            sample = self.nusc.get('sample', sample_token)
            annotations.extend(sample['anns'])
            sample_token = sample['next']

        unique_instances = set(
            self.nusc.get('sample_annotation', ann)['instance_token'] for ann in annotations
        )

        return [
            InstanceFrame(self.nusc, token, self.scene_index, self.scene_ego_positions, self.ego_start_time)
            for token in unique_instances
        ]

    def query_exist(self, obj, direction="ahead"):
        return any(
            obj in inst.category_name and getattr(self, f"is_{direction}")(inst)
            for inst in self.instances
        )

    def query_count(self, obj, direction=None):
        objs = [inst for inst in self.instances if obj in inst.category_name]
        if direction:
            objs = [inst for inst in objs if getattr(self, f"is_{direction}")(inst)]
        return len(objs)

    @staticmethod
    def is_ahead(inst):
        return np.mean([loc[1] for loc in inst.locations]) > 0

    @staticmethod
    def is_behind(inst):
        return np.mean([loc[1] for loc in inst.locations]) < 0

    @staticmethod
    def is_left(inst):
        return np.mean([loc[0] for loc in inst.locations]) < 0

    @staticmethod
    def is_right(inst):
        return np.mean([loc[0] for loc in inst.locations]) > 0

# --------------------------
# Scene Summary
# --------------------------
def summarize_scene(scene_frame):
    return " ".join([inst.describe_movement() for inst in scene_frame.instances])

# --------------------------
# VQA Answering
# --------------------------
def answer(scene_frame, question):
    q = question.lower()
    if "passenger" in q:
        return "Passenger information is not available in nuScenes dataset."

    if "is there" in q:
        obj = q.split("is there a ")[-1].split()[0]
        direction = next((d for d in ["ahead", "behind", "left", "right"] if d in q), "ahead")
        return "Yes" if scene_frame.query_exist(obj, direction) else "No"

    elif "how many" in q:
        obj = q.split("how many ")[-1].split()[0]
        direction = next((d for d in ["ahead", "behind", "left", "right"] if d in q), None)
        return scene_frame.query_count(obj, direction)

    elif "what objects" in q or "what is" in q:
        objs = set(inst.category_name for inst in scene_frame.instances)
        return ", ".join(objs)

    elif "which is closer" in q:
        tokens = q.replace("which is closer, ", "").split(" or ")
        obj1, obj2 = tokens[0], tokens[1]
        dist1 = min(
            [np.mean([np.linalg.norm(loc) for loc in inst.locations])
             for inst in scene_frame.instances if obj1 in inst.category_name] or [float('inf')]
        )
        dist2 = min(
            [np.mean([np.linalg.norm(loc) for loc in inst.locations])
             for inst in scene_frame.instances if obj2 in inst.category_name] or [float('inf')]
        )
        return obj1 if dist1 < dist2 else obj2

    elif "describe the scene" in q:
        return summarize_scene(scene_frame)

    else:
        return "Question type not supported."
