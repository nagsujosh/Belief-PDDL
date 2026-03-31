from typing import List, Tuple


PREDICATE_ORDER = [
    "arm_empty()",
    "clear(block_0)",
    "clear(block_1)",
    "clear(block_2)",
    "on_table(block_0)",
    "on_table(block_1)",
    "on_table(block_2)",
    "holding(block_0)",
    "holding(block_1)",
    "holding(block_2)",
    "visible(block_0)",
    "visible(block_1)",
    "visible(block_2)",
    "on(block_0,block_1)",
    "on(block_0,block_2)",
    "on(block_1,block_0)",
    "on(block_1,block_2)",
    "on(block_2,block_0)",
    "on(block_2,block_1)",
]


def semantic_queries(predicate: str) -> Tuple[str, str]:
    colors = {"block_0": "red", "block_1": "blue", "block_2": "green", "block_3": "yellow", "block_4": "purple"}
    if predicate.startswith("clear("):
        b = predicate[6:-1]
        c = colors.get(b)
        return f"A {c} block with no blocks on top of it.", f"A {c} block with another block resting on top of it."
    if predicate.startswith("on_table("):
        b = predicate[9:-1]
        c = colors.get(b)
        return f"A {c} block touching the grey table.", f"A {c} block stacked high above the grey table."
    if predicate.startswith("holding("):
        b = predicate[8:-1]
        c = colors.get(b)
        return f"A {c} block floating in the air.", f"A {c} block resting still down on the stack."
    if predicate.startswith("visible("):
        b = predicate[8:-1]
        c = colors.get(b)
        return f"A {c} block.", f"There is no {c} block."
    if predicate == "arm_empty()":
        return "No blocks are floating.", "A block is floating."
    if predicate.startswith("on("):
        b1, b2 = predicate[3:-1].split(",")
        c1 = colors.get(b1.strip())
        c2 = colors.get(b2.strip())
        return f"A {c1} block exactly on top of a {c2} block.", f"A {c1} block is NOT touching the {c2} block."
    return predicate, ""


def grounded_predicates(blocks: List[str]) -> List[str]:
    preds = ["arm_empty()"]
    for b1 in blocks:
        preds.extend([f"clear({b1})", f"on_table({b1})", f"holding({b1})", f"visible({b1})"])
        for b2 in blocks:
            if b1 != b2:
                preds.append(f"on({b1},{b2})")
    return preds
