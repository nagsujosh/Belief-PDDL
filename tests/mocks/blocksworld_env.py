import numpy as np
from PIL import Image, ImageDraw
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class Observation:
    rgb: np.ndarray
    visible_objects: List[str]
    gt_predicates: List[str]

class MockBlocksworldEnv:
    def __init__(self, num_blocks=3):
        self.num_blocks = num_blocks
        self.blocks = [f"block_{i}" for i in range(num_blocks)]
        self.reset()
        
    def reset(self):
        import random
        # Create a procedural randomized state
        self.state = {
            "on_table": set(),
            "on": {}, # child -> parent
            "clear": set(),
            "holding": None,
            "arm_empty": True,
            "visible": set()
        }
        
        # Shuffle blocks to create random towers
        shuffled = list(self.blocks)
        random.shuffle(shuffled)
        
        # Build towers
        remaining = set(shuffled)
        while remaining:
            # Pick a random tower height between 1 and remaining
            h = random.randint(1, len(remaining))
            tower = [remaining.pop() for _ in range(h)]
            
            # Bottom block is on table
            self.state["on_table"].add(tower[0])
            # Stack the rest
            for i in range(1, len(tower)):
                self.state["on"][tower[i]] = tower[i-1]
            # Top block is clear
            self.state["clear"].add(tower[-1])
            
        # Mathematical Occlusion Mock: 
        # For our tests, say 30% of blocks that are strictly underneath other blocks are 'hidden' from view
        # and 30% of standard blocks are manually blocked from camera angle.
        all_blocks = set(self.blocks)
        hidden = set()
        for b in all_blocks:
            if random.random() < 0.3:
                hidden.add(b)
                
        self.state["visible"] = all_blocks - hidden
        
        return self._get_obs()
        
    def _render(self) -> np.ndarray:
        # Mock simple render based on state
        img = Image.new("RGB", (256, 256), color="white")
        draw = ImageDraw.Draw(img)
        
        # Ground
        draw.rectangle([(0, 200), (256, 256)], fill="grey")
        
        # Draw some arbitrary boxes for visible blocks
        x = 50
        for b in self.state["visible"]:
            draw.rectangle([(x, 150), (x+40, 190)], fill="blue", outline="black")
            draw.text((x+5, 165), b[-1], fill="white")
            x += 60
            
        return np.array(img)

    def _get_gt_predicates(self) -> List[str]:
        preds = []
        if self.state["arm_empty"]:
            preds.append("arm_empty()")
        if self.state["holding"]:
            preds.append(f"holding({self.state['holding']})")
        
        for b in self.state["on_table"]:
            preds.append(f"on_table({b})")
            
        for child, parent in self.state["on"].items():
            preds.append(f"on({child},{parent})")
            
        for b in self.state["clear"]:
            preds.append(f"clear({b})")
            
        for b in self.state["visible"]:
            preds.append(f"visible({b})")
            
        return preds

    def _get_obs(self) -> Observation:
        return Observation(
            rgb=self._render(),
            visible_objects=list(self.state["visible"]),
            gt_predicates=self._get_gt_predicates()
        )

    def step(self, action: str, args: List[str]) -> Tuple[Observation, float, bool]:
        if action == "reveal_side":
            # Mock sensing action: make target randomly visible
            if args[0] in self.blocks:
                self.state["visible"].add(args[0])
                
        elif action == "pickup":
            x = args[0]
            if x in self.state["clear"] and x in self.state["on_table"] and self.state["arm_empty"] and x in self.state["visible"]:
                self.state["on_table"].remove(x)
                self.state["clear"].remove(x)
                self.state["arm_empty"] = False
                self.state["holding"] = x
                
        elif action == "putdown":
            x = args[0]
            if self.state["holding"] == x:
                self.state["holding"] = None
                self.state["arm_empty"] = True
                self.state["clear"].add(x)
                self.state["on_table"].add(x)

        elif action == "stack":
            x = args[0]
            y = args[1]
            if self.state["holding"] == x and y in self.state["clear"] and y in self.state["visible"]:
                self.state["holding"] = None
                self.state["arm_empty"] = True
                self.state["clear"].remove(y)
                self.state["clear"].add(x)
                self.state["on"][x] = y
                
        elif action == "unstack":
            x = args[0]
            y = args[1]
            if x in self.state["clear"] and self.state["on"].get(x) == y and self.state["arm_empty"] and x in self.state["visible"]:
                self.state["clear"].remove(x)
                self.state["holding"] = x
                self.state["arm_empty"] = False
                del self.state["on"][x]
                self.state["clear"].add(y)
                
        # To avoid making this too long, simplistic task condition: target is to have block 0 and block 1 stacked
        # Just check if reward condition met
        reward = 1.0 if ("on" in self.state and self.state["on"].get("block_0") == "block_1") else 0.0
        done = reward > 0
        return self._get_obs(), reward, done
