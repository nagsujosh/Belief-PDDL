import torch
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.perception.predicate_queries import PredicateQueryBuilder
from src.perception.crop_builder import PerceptionCropBuilder
from src.perception.backbones import VisionBackbone, DummyTextEncoder
from src.perception.unary_head import UnaryPredicateHead
from src.perception.binary_head import BinaryPredicateHead
from src.perception.calibrate import TemperatureScalar

def test_perception_pipeline():
    print("Testing perception shapes...")

    # 1. Test query builder
    config = {
        "unary": [{"name": "visible"}, {"name": "on_table"}],
        "binary": [{"name": "on"}, {"name": "holding"}]
    }
    builder = PredicateQueryBuilder(config)
    objects = ["b0", "b1", "b2"]
    
    unary_q = builder.build_unary_queries(objects)
    assert len(unary_q) == 6, f"Expected 6 unary queries, got {len(unary_q)}"
    
    binary_q = builder.build_binary_queries(objects)
    assert len(binary_q) == 12, f"Expected 12 pairwise permutations * 2 predicates, got {len(binary_q)}"
    
    print("✅ Predicate Queries constructed properly.")

    # 2. Test Crops
    dummy_img = Image.new("RGB", (256, 256), color="red")
    crop_builder = PerceptionCropBuilder(target_size=(224, 224))
    crop_u = crop_builder.extract_unary_crop(dummy_img, "b0")
    assert crop_u.shape == (3, 224, 224), "Crop shape error"
    
    crop_b1, crop_b2, crop_union = crop_builder.extract_binary_crops(dummy_img, "b0", "b1")
    assert crop_b1.shape == (3, 224, 224), "Crop shapes error"

    print("✅ Image Cropping outputs accurate tensor domains.")

    # 3. Test Backbone and Heads integration
    img_tensor = crop_u.unsqueeze(0) # Batch 1
    vision_model = VisionBackbone(name="resnet18", pretrained=False, out_features=256)
    text_model = DummyTextEncoder(out_features=256)
    
    v_feat = vision_model(img_tensor) # (1, 256)
    t_feat = text_model(["visible"])  # (1, 256)
    
    assert v_feat.shape == (1, 256), "Vision embedding misshape."
    
    unary_head = UnaryPredicateHead(visual_dim=256, text_dim=256)
    logit = unary_head(v_feat, t_feat)
    assert logit.shape == (1, 1), "Logits shape out of bounds."

    # Binary pass
    binary_head = BinaryPredicateHead(visual_dim=256, text_dim=256, geom_dim=4)
    v_b1 = vision_model(crop_b1.unsqueeze(0))
    v_b2 = vision_model(crop_b2.unsqueeze(0))
    v_union = vision_model(crop_union.unsqueeze(0))
    rel_geom = torch.randn(1, 4)
    b_logit = binary_head(v_b1, v_b2, v_union, t_feat, rel_geom)
    assert b_logit.shape == (1, 1), "Binary logits shape out of bounds."

    print("✅ Deep ResNet Backbone integration emits correct Logit distributions via Neural Heads.")

    # 4. Test calibrator
    scalar = TemperatureScalar(init_temp=1.5)
    prob = scalar(logit)
    assert 0 <= prob.item() <= 1, "Probability exceeded limits."
    
    print("✅ Calibrator maps logits successfully.")
    print("🎉 Phase 2 Pipeline verified successfully.")

if __name__ == "__main__":
    test_perception_pipeline()
