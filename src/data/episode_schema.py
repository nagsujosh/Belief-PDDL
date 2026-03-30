import json
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class ObjectMeta:
    id: str
    type: str

@dataclass
class EpisodeStep:
    t: int
    rgb_path: str
    depth_path: Optional[str]
    visible_objects: List[str]
    gt_predicates: List[str]
    action: Optional[str]
    reward: float
    done: bool

@dataclass
class EpisodeTrajectory:
    episode_id: str
    domain: str
    task_text: str
    objects: List[ObjectMeta]
    steps: List[EpisodeStep]

    def to_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
