import os
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

@dataclass
class ALFWorldObservation:
    rgb: np.ndarray
    visible_objects: List[str]
    gt_predicates: List[str]
    text_feedback: str
    won: bool = False
    admissible_commands: Optional[List[str]] = None
    gamefile: Optional[str] = None

class ALFWorldEnvWrapper:
    def __init__(
        self,
        task_config: Optional[str] = None,
        task_type_id: int = 1,
        data_path: Optional[str] = None,
        object_target: Optional[str] = None,
        parent_target: Optional[str] = None,
        train_eval: str = "eval_in_distribution",
        max_episode_steps: int = 50,
        num_eval_games: int = 1,
    ):
        """
        Bridging Wrapper for ALFWorld.
        Prefers AlfredTWEnv so we can exercise the planning stack in text mode
        without depending on AI2-THOR visual extras.
        """
        self.task_config = task_config
        self.task_type_id = task_type_id
        self.data_path = data_path
        self.object_target = object_target.lower() if object_target else None
        self.parent_target = parent_target.lower() if parent_target else None
        self.train_eval = train_eval
        self.max_episode_steps = max_episode_steps
        self.num_eval_games = num_eval_games
        
        try:
            import alfworld.agents.environment as environment
            import alfworld.agents.modules.generic as generic
            from alfworld import info as alfworld_info
            self.config = self._load_or_build_config(generic, alfworld_info)
            env_type = self.config["env"]["type"]
            env_cls = environment.get_environment(env_type)
            self.env_driver = env_cls(self.config, train_eval=self.train_eval)
            self._filter_game_files()
            self.env = self.env_driver.init_env(batch_size=1)
        except ImportError:
            raise ImportError(
                "ALFWorld is not available in this environment. "
                "Install `alfworld` for text mode, and `alfworld[vis]` if you also want AI2-THOR visual support."
            )
            
        self.objects = []
        self.state = {
            "at": "countertop_1",
            "in": {},
            "on": {},
            "open": set(),
            "closed": set(),
            "holding": None,
            "visible": set()
        }
        self.reset()

    def _filter_game_files(self):
        if not hasattr(self.env_driver, "game_files"):
            return
        if not self.object_target and not self.parent_target:
            return

        filtered = []
        for game_file in self.env_driver.game_files:
            traj_path = Path(game_file).with_name("traj_data.json")
            if not traj_path.exists():
                continue
            with traj_path.open() as handle:
                traj = json.load(handle)
            pddl = traj.get("pddl_params", {})
            obj = str(pddl.get("object_target", "")).lower()
            parent = str(pddl.get("parent_target", "")).lower()
            if self.object_target and obj != self.object_target:
                continue
            if self.parent_target and parent != self.parent_target:
                continue
            filtered.append(game_file)

        if not filtered:
            raise FileNotFoundError(
                "No ALFWorld games matched the requested task filter. "
                f"object_target={self.object_target!r}, parent_target={self.parent_target!r}"
            )

        if self.num_eval_games > 0:
            filtered = filtered[:self.num_eval_games]

        self.env_driver.game_files = filtered
        self.env_driver.num_games = len(filtered)

    def _load_or_build_config(self, generic, alfworld_info):
        config_path = Path(self.task_config).expanduser() if self.task_config else None
        if config_path:
            if not config_path.exists():
                raise FileNotFoundError(
                    f"ALFWorld config file not found: {config_path}. "
                    "Either provide a valid --task-config path or omit it and pass --alfworld-data "
                    "to auto-build a text-mode AlfredTWEnv config."
                )
            original_argv = list(sys.argv)
            try:
                # ALFWorld's config loader parses sys.argv directly. Shield it from
                # this benchmark's CLI flags and pass only the requested config file.
                sys.argv = [original_argv[0], str(config_path)]
                return generic.load_config()
            finally:
                sys.argv = original_argv

        resolved_data_path = self._resolve_data_path(alfworld_info)
        return {
            "env": {
                "type": "AlfredTWEnv",
                "goal_desc_human_anns_prob": 0.0,
                "task_types": [self.task_type_id],
                "domain_randomization": False,
                "expert_type": "handcoded",
            },
            "dataset": {
                "data_path": resolved_data_path,
                "eval_id_data_path": resolved_data_path,
                "eval_ood_data_path": resolved_data_path,
                "num_train_games": 0,
                "num_eval_games": 0,
            },
            "logic": {
                "domain": alfworld_info.ALFRED_PDDL_PATH,
                "grammar": alfworld_info.ALFRED_TWL2_PATH,
            },
            "general": {
                "training_method": "dqn",
            },
            "rl": {
                "training": {
                    "max_nb_steps_per_episode": self.max_episode_steps,
                }
            },
        }

    def _resolve_data_path(self, alfworld_info) -> str:
        candidates = []
        if self.data_path:
            candidates.append(Path(self.data_path).expanduser())
        env_data = os.environ.get("ALFWORLD_DATA")
        if env_data:
            candidates.append(Path(env_data).expanduser())
        candidates.append(Path(alfworld_info.ALFWORLD_DATA).expanduser())

        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if self._looks_like_alfworld_data(candidate):
                return str(candidate)

        looked = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Could not find an ALFWorld text-game dataset. "
            "Pass --alfworld-data <dataset_root> or set ALFWORLD_DATA to a directory "
            "containing generated ALFWorld games with traj_data.json and game.tw-pddl. "
            f"Checked: {looked}"
        )

    def _looks_like_alfworld_data(self, root: Path) -> bool:
        if not root.exists() or not root.is_dir():
            return False
        for current_root, _, files in os.walk(root):
            if "traj_data.json" in files and "game.tw-pddl" in files:
                return True
        return False
        
    def reset(self) -> ALFWorldObservation:
        obs, info = self.env.reset()
        return self._wrap_env_output(obs, info)

    def _render(self) -> np.ndarray:
        # Mocking real Image return for pipeline continuity
        return np.zeros((300, 300, 3), dtype=np.uint8)

    def _get_gt_predicates(self) -> List[str]:
        preds = []
        preds.append(f"at(agent,{self.state['at']})")
        
        for k, v in self.state["in"].items():
            preds.append(f"in({k},{v})")
        for k, v in self.state["on"].items():
            preds.append(f"on({k},{v})")
            
        for r in self.state["open"]: preds.append(f"open({r})")
        for r in self.state["closed"]: preds.append(f"closed({r})")
        
        if self.state["holding"]:
            preds.append(f"holding(agent,{self.state['holding']})")
            
        for b in self.state["visible"]:
            preds.append(f"visible({b})")
            
        return preds

    def _get_obs(
        self,
        text: str,
        won: bool = False,
        admissible_commands: Optional[List[str]] = None,
        gamefile: Optional[str] = None,
    ) -> ALFWorldObservation:
        return ALFWorldObservation(
            rgb=self._render(),
            visible_objects=list(self.state["visible"]),
            gt_predicates=self._get_gt_predicates(),
            text_feedback=text,
            won=won,
            admissible_commands=admissible_commands,
            gamefile=gamefile,
        )

    def _wrap_env_output(self, obs, info) -> ALFWorldObservation:
        text = obs[0]
        won = bool(info.get("won", [False])[0]) if isinstance(info, dict) else False
        admissible = info.get("admissible_commands", [[]])[0] if isinstance(info, dict) else []
        gamefile = info.get("extra.gamefile", [None])[0] if isinstance(info, dict) else None
        self._live_cache = text
        self._live_admissible = list(admissible)
        return self._get_obs(
            text=text,
            won=won,
            admissible_commands=admissible,
            gamefile=gamefile,
        )

    def _to_alf_name(self, symbol: str) -> str:
        return " ".join(symbol.split("_"))

    def _resolve_admissible_command(self, action: str, args: List[str]) -> Optional[str]:
        commands = [cmd.strip() for cmd in getattr(self, "_live_admissible", []) if cmd]
        lowered = [cmd.lower() for cmd in commands]
        if not commands:
            return None

        if action in {"take_from_container", "take_from_surface"} and len(args) >= 2:
            obj_phrase = self._to_alf_name(args[0]).lower()
            recep_phrase = self._to_alf_name(args[1]).lower()
            matches = [
                cmd for cmd, low in zip(commands, lowered)
                if low.startswith("take ")
                and f" from {recep_phrase}" in low
                and low[len("take "):].startswith(obj_phrase)
            ]
            if matches:
                return sorted(matches, key=len)[0]

        if action == "put_in_container" and len(args) >= 2:
            obj_phrase = self._to_alf_name(args[0]).lower()
            recep_phrase = self._to_alf_name(args[1]).lower()
            matches = [
                cmd for cmd, low in zip(commands, lowered)
                if (
                    (low.startswith("move ") and f" to {recep_phrase}" in low)
                    or (low.startswith("put ") and f" in {recep_phrase}" in low)
                )
                and low.split(" ", 1)[1].startswith(obj_phrase)
            ]
            if matches:
                return sorted(matches, key=len)[0]

        if action == "put_on_surface" and len(args) >= 2:
            obj_phrase = self._to_alf_name(args[0]).lower()
            recep_phrase = self._to_alf_name(args[1]).lower()
            matches = [
                cmd for cmd, low in zip(commands, lowered)
                if (
                    (low.startswith("move ") and f" to {recep_phrase}" in low)
                    or (low.startswith("put ") and f" on {recep_phrase}" in low)
                )
                and low.split(" ", 1)[1].startswith(obj_phrase)
            ]
            if matches:
                return sorted(matches, key=len)[0]

        return None

    def step(self, action: str, args: List[str]) -> Tuple[ALFWorldObservation, float, bool]:
        """
        Translates PDDL determinist strings directly down to embodied textual actions.
        """
        if args and args[0] == "agent":
            args = args[1:]

        resolved = self._resolve_admissible_command(action, args)
        if resolved is not None:
            alf_cmd = resolved
        elif action == "goto_location":
            alf_cmd = f"go to {self._to_alf_name(args[-1])}"
        elif action == "open_receptacle":
            alf_cmd = f"open {self._to_alf_name(args[-1])}"
        elif action == "close_receptacle":
            alf_cmd = f"close {self._to_alf_name(args[-1])}"
        elif action == "take_from_container":
            alf_cmd = f"take {self._to_alf_name(args[0])} from {self._to_alf_name(args[1])}"
        elif action == "take_from_surface":
            alf_cmd = f"take {self._to_alf_name(args[0])} from {self._to_alf_name(args[1])}"
        elif action == "put_in_container":
            alf_cmd = f"put {self._to_alf_name(args[0])} in {self._to_alf_name(args[1])}"
        elif action == "put_on_surface":
            alf_cmd = f"put {self._to_alf_name(args[0])} on {self._to_alf_name(args[1])}"
        elif action == "look_inside":
            alf_cmd = f"look in {self._to_alf_name(args[0])}"
        else:
            alf_cmd = "look"

        obs, scores, dones, infos = self.env.step([alf_cmd])
        return self._wrap_env_output(obs, infos), float(scores[0]), bool(dones[0])
