import itertools
from typing import List, Dict, Tuple, Any

class PredicateQueryBuilder:
    def __init__(self, predicate_config: Dict[str, Any]):
        self.config = predicate_config
        self.unary_predicates = [p['name'] for p in self.config.get('unary', [])]
        self.binary_predicates = [p['name'] for p in self.config.get('binary', [])]

    def build_unary_queries(self, visible_objects: List[str]) -> List[Tuple[str, str]]:
        """
        Builds queries of the form (predicate_name, object_id)
        Returns: list of (pred_name, obj)
        """
        queries = []
        for obj in visible_objects:
            for pred in self.unary_predicates:
                queries.append((pred, obj))
        return queries

    def build_binary_queries(self, visible_objects: List[str]) -> List[Tuple[str, str, str]]:
        """
        Builds queries of the form (predicate_name, object_a_id, object_b_id)
        Returns: list of (pred_name, obj_a, obj_b)
        """
        queries = []
        # Generate all permutations of length 2
        pairs = itertools.permutations(visible_objects, 2)
        for obj_a, obj_b in pairs:
            for pred in self.binary_predicates:
                queries.append((pred, obj_a, obj_b))
        return queries
