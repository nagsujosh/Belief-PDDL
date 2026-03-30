import sys
import os
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.blocksworld_env import MockBlocksworldEnv
from src.perception.predicate_queries import PredicateQueryBuilder
from src.perception.crop_builder import PerceptionCropBuilder
from src.perception.backbones import VisionBackbone, DummyTextEncoder
from src.perception.unary_head import UnaryPredicateHead
from src.perception.calibrate import TemperatureScalar

def demo_pipeline():
    print("\n--- 🧱 PHASE 1: ENVIRONMENT DEMO ---")
    print("Instantiating Mock Blocksworld...")
    env = MockBlocksworldEnv(num_blocks=3)
    
    print("\nTaking an action to reveal 'block_2'...")
    obs, reward, done = env.step("reveal_side", ["block_2"])
    
    print(f"Visible Objects: {obs.visible_objects}")
    print(f"Ground Truth Predicates (Sample): {obs.gt_predicates[:4]}...")
    
    # Save image so user can see what it generated
    os.makedirs("outputs/plots", exist_ok=True)
    img_path = "outputs/plots/demo_obs.png"
    Image.fromarray(obs.rgb).save(img_path)
    print(f"Saved generated generic observation image to: {img_path}")
    
    print("\n--- 🧠 PHASE 2: PERCEPTION DEMO ---")
    config = {
        "unary": [{"name": "visible"}, {"name": "on_table"}],
        "binary": [] # skip binary for brevity
    }
    
    print("1. Building Queries for objects...")
    builder = PredicateQueryBuilder(config)
    unary_queries = builder.build_unary_queries(obs.visible_objects)
    print(f"Generated queries: {unary_queries[:3]}...")
    
    print("\n2. Extracting image crops...")
    img = Image.fromarray(obs.rgb)
    crop_builder = PerceptionCropBuilder()
    
    # Let's just process the first query
    pred_name, obj_id = unary_queries[0]
    print(f"Processing query: '{pred_name}({obj_id})'")
    
    crop = crop_builder.extract_unary_crop(img, obj_id)
    # Add batch dimension
    crop_batch = crop.unsqueeze(0)
    
    print("\n3. Running Neural Networks...")
    vision_model = VisionBackbone(name="resnet18")
    text_model = DummyTextEncoder(out_features=256)
    unary_head = UnaryPredicateHead(visual_dim=256, text_dim=256)
    calibrator = TemperatureScalar(init_temp=1.5)
    
    # Pass through models
    v_feat = vision_model(crop_batch)
    t_feat = text_model([pred_name])
    logit = unary_head(v_feat, t_feat)
    
    print(f"Raw Neural Logit Output: {logit.item():.4f}")
    
    prob = calibrator(logit)
    print(f"Calibrated Belief Probability: {prob.item():.4f} (Between 0 and 1)")
    
    print("\n✅ Verification Complete! The pure simulation links flawlessly with the deep learning tensor models.")

if __name__ == "__main__":
    # Ensure reproducibility of random dummy weights
    torch.manual_seed(42)
    demo_pipeline()
