import re
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple


def symbol_to_phrase(symbol: str) -> str:
    return symbol.replace("_", " ").lower()


def phrase_variants(symbol: str) -> List[str]:
    base = symbol_to_phrase(symbol)
    compact = base.replace(" ", "")
    return [base, compact]


OPENABLE_RECEPTACLE_PREFIXES = (
    "fridge",
    "cabinet",
    "drawer",
    "microwave",
    "garbagecan",
)


def canonical_symbol(text: str) -> str:
    return re.sub(r"\s+", "_", text.strip().lower())


@dataclass
class ALFWorldTaskSpec:
    name: str
    goal: str
    task_type_id: int
    start_receptacle: str
    objects: List[str]
    containers: List[str]
    surfaces: List[str]

    @property
    def receptacles(self) -> List[str]:
        return self.containers + self.surfaces

    @property
    def planner_objects(self) -> List[str]:
        return ["agent"] + self.objects + self.receptacles

    @property
    def typed_objects(self) -> Dict[str, List[str]]:
        return {
            "agent": ["agent"],
            "item": list(self.objects),
            "container": list(self.containers),
            "surface": list(self.surfaces),
        }

    def with_receptacles(self, containers: Sequence[str], surfaces: Sequence[str]) -> "ALFWorldTaskSpec":
        return replace(self, containers=list(containers), surfaces=list(surfaces))


TASK_PRESETS: Dict[str, ALFWorldTaskSpec] = {
    "put_egg_in_microwave": ALFWorldTaskSpec(
        name="put_egg_in_microwave",
        goal="(in egg microwave_1)",
        task_type_id=1,
        start_receptacle="",
        objects=["egg"],
        containers=["microwave_1"],
        surfaces=[],
    ),
    "put_glassbottle_in_fridge": ALFWorldTaskSpec(
        name="put_glassbottle_in_fridge",
        goal="(in glassbottle fridge_1)",
        task_type_id=1,
        start_receptacle="",
        objects=["glassbottle"],
        containers=["fridge_1"],
        surfaces=[],
    ),
    "put_apple_in_fridge": ALFWorldTaskSpec(
        name="put_apple_in_fridge",
        goal="(in apple fridge_1)",
        task_type_id=1,
        start_receptacle="",
        objects=["apple"],
        containers=["fridge_1"],
        surfaces=[],
    ),
}


def get_task_spec(name: str) -> ALFWorldTaskSpec:
    if name not in TASK_PRESETS:
        raise ValueError(f"Unknown ALFWorld task preset: {name}")
    return TASK_PRESETS[name]


def receptacles_from_admissible_commands(commands: Sequence[str]) -> Tuple[List[str], List[str]]:
    receptacles: List[str] = []
    for command in commands:
        lowered = command.lower().strip()
        if lowered.startswith("go to "):
            receptacles.append(canonical_symbol(lowered[len("go to "):]))

    unique = sorted(set(receptacles))
    containers = [name for name in unique if any(name.startswith(prefix) for prefix in OPENABLE_RECEPTACLE_PREFIXES)]
    surfaces = [name for name in unique if name not in containers]
    return containers, surfaces


NEGATIVE_FEEDBACK_PATTERNS = (
    "nothing happens",
    "you can't",
    "cannot",
    "don't see",
    "do not see",
    "not sure what",
    "there is no",
    "you are not",
    "you need to",
    "is closed",
)


def feedback_indicates_success(text: str) -> bool:
    lowered = text.lower()
    return not any(pattern in lowered for pattern in NEGATIVE_FEEDBACK_PATTERNS)


def action_feedback_success(action: Optional[str], text: str) -> bool:
    lowered = text.lower()
    if action == "goto_location":
        return lowered.startswith("you arrive at ")
    if action == "open_receptacle":
        return "you open" in lowered
    if action == "close_receptacle":
        return "you close" in lowered
    if action in {"take_from_container", "take_from_surface"}:
        return "you pick up" in lowered or "you take" in lowered
    if action in {"put_in_container", "put_on_surface"}:
        return "you put" in lowered or "you place" in lowered
    return feedback_indicates_success(text)


def extract_visible_entities(text: str, candidates: Sequence[str]) -> List[str]:
    lowered = text.lower()
    visible = []
    for symbol in candidates:
        if any(variant in lowered for variant in phrase_variants(symbol)):
            visible.append(symbol)
    return visible


def extract_open_closed(text: str, containers: Sequence[str]) -> Tuple[List[str], List[str]]:
    lowered = text.lower()
    opened, closed = [], []
    for container in containers:
        phrase = symbol_to_phrase(container)
        if f"{phrase} is open" in lowered or f"open {phrase}" in lowered:
            opened.append(container)
        if f"{phrase} is closed" in lowered or f"close {phrase}" in lowered:
            closed.append(container)
    return opened, closed


def parse_action_effect(
    action: Optional[str],
    args: Sequence[str],
    text: str,
) -> Dict[str, Tuple[str, ...]]:
    if not action or not action_feedback_success(action, text):
        return {}

    norm_args = list(args)
    if norm_args and norm_args[0] == "agent":
        norm_args = norm_args[1:]

    if action == "goto_location" and len(norm_args) >= 1:
        return {"at": ("agent", norm_args[-1])}
    if action == "open_receptacle" and norm_args:
        return {"open": (norm_args[-1],)}
    if action == "close_receptacle" and norm_args:
        return {"closed": (norm_args[-1],)}
    if action == "take_from_container" and len(norm_args) >= 2:
        return {"holding": ("agent", norm_args[0])}
    if action == "take_from_surface" and len(norm_args) >= 2:
        return {"holding": ("agent", norm_args[0])}
    if action == "put_in_container" and len(norm_args) >= 2:
        return {"in": (norm_args[0], norm_args[1])}
    if action == "put_on_surface" and len(norm_args) >= 2:
        return {"on": (norm_args[0], norm_args[1])}
    return {}


def observation_evidence(
    text: str,
    task_spec: ALFWorldTaskSpec,
) -> Dict[str, bool]:
    evidence: Dict[str, bool] = {}
    visible_entities = extract_visible_entities(text, task_spec.objects + task_spec.receptacles)
    opened, closed = extract_open_closed(text, task_spec.containers)

    for entity in visible_entities:
        if entity in task_spec.objects:
            evidence[f"visible({entity})"] = True

    for container in opened:
        evidence[f"open({container})"] = True
        evidence[f"closed({container})"] = False
    for container in closed:
        evidence[f"closed({container})"] = True
        evidence[f"open({container})"] = False

    return evidence


def negative_location_evidence(
    text: str,
    task_spec: ALFWorldTaskSpec,
    location: Optional[str] = None,
) -> Dict[str, bool]:
    lowered = text.lower()
    evidence: Dict[str, bool] = {}
    if not location:
        return evidence

    phrase = symbol_to_phrase(location)
    target = task_spec.objects[0] if task_spec.objects else None
    if not target:
        return evidence
    target_visible = bool(extract_visible_entities(text, [target]))

    if location in task_spec.surfaces:
        if f"on the {phrase}, you see nothing" in lowered:
            evidence[f"on({target},{location})"] = False
            evidence[f"visible({target})"] = False
        elif f"on the {phrase}, you see" in lowered and not target_visible:
            evidence[f"on({target},{location})"] = False
            evidence[f"visible({target})"] = False

    if location in task_spec.containers:
        if (
            f"the {phrase} is open" in lowered
            and "you see nothing" in lowered
        ):
            evidence[f"in({target},{location})"] = False
            evidence[f"visible({target})"] = False
        elif (
            f"the {phrase} is open" in lowered
            and "you see" in lowered
            and not target_visible
        ):
            evidence[f"in({target},{location})"] = False
            evidence[f"visible({target})"] = False

    return evidence
