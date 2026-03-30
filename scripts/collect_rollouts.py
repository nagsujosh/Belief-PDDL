import sys
import os
import uuid
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.blocksworld_env import MockBlocksworldEnv
from src.data.episode_schema import EpisodeTrajectory, EpisodeStep, ObjectMeta

def run_scripted_rollout():
    env = MockBlocksworldEnv(num_blocks=3)
    obs = env.reset()
    
    episode_id = f"bw_{uuid.uuid4().hex[:6]}"
    out_dir = os.path.join("data", "raw", episode_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Start State
    steps = []
    
    # Save image
    img_path = os.path.join(out_dir, "step_0.png")
    Image.fromarray(obs.rgb).save(img_path)
    
    steps.append(EpisodeStep(
        t=0,
        rgb_path=img_path,
        depth_path=None,
        visible_objects=obs.visible_objects.copy(),
        gt_predicates=obs.gt_predicates.copy(),
        action=None,
        reward=0.0,
        done=False
    ))
    
    # 2. Sensing action
    obs, reward, done = env.step("reveal_side", ["block_2"])
    
    img_path = os.path.join(out_dir, "step_1.png")
    Image.fromarray(obs.rgb).save(img_path)
    
    steps.append(EpisodeStep(
        t=1,
        rgb_path=img_path,
        depth_path=None,
        visible_objects=obs.visible_objects.copy(),
        gt_predicates=obs.gt_predicates.copy(),
        action="reveal_side(block_2)",
        reward=reward,
        done=done
    ))
    
    # 3. Pickup action on block 0
    obs, reward, done = env.step("pickup", ["block_0"])
    
    img_path = os.path.join(out_dir, "step_2.png")
    Image.fromarray(obs.rgb).save(img_path)
    
    steps.append(EpisodeStep(
        t=2,
        rgb_path=img_path,
        depth_path=None,
        visible_objects=obs.visible_objects.copy(),
        gt_predicates=obs.gt_predicates.copy(),
        action="pickup(block_0)",
        reward=reward,
        done=done
    ))
    
    # Create the top level trajectory
    traj = EpisodeTrajectory(
        episode_id=episode_id,
        domain="blocksworld",
        task_text="put block_0 on block_1",
        objects=[ObjectMeta(id=f"block_{i}", type="block") for i in range(3)],
        steps=steps
    )
    
    out_json = os.path.join(out_dir, "episode.json")
    traj.to_json(out_json)
    
    print(f"✅ Successfully collected rollout: {episode_id}")
    print(f"   Outputs saved to: {out_dir}")

if __name__ == "__main__":
    run_scripted_rollout()
