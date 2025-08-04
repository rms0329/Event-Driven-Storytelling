import re


class UnavailableRelationshipError(Exception):
    pass


class UnavailableAnchorError(Exception):
    pass


_AVAILABLE_RELATIONSHIPS = [
    "close to",
    "adjacent to",
    "interact with",
    "inside",
    "sit on",
    "lie on",
    "to the left of",
    "to the right of",
    "in front of",
    "behind",
    "between",
    "aligned with",
    "look at",
]


class RelationshipEvaluator:
    def __init__(self):
        self.available_relationships = re.compile(r"^(" + "|".join(_AVAILABLE_RELATIONSHIPS) + ")")
        self.characters = []
        self.objects = []

    def reset_available_anchors(self, characters, objects):
        self.characters = characters
        self.objects = objects

    def evaluate(self, condition_str, relationships):
        if not all(self.available_relationships.match(r) for r in relationships):
            raise UnavailableRelationshipError(f"Unavailable relationships: {relationships}")

        if not condition_str:
            return True

        tokens = re.split(r"(\|\||&&|\(|\))", condition_str)
        tokens = [t.strip() for t in tokens if t.strip() != ""]
        value, index = self._parse_expression(tokens, 0, relationships)
        if index != len(tokens):
            raise SyntaxError("Unexpected tokens at the end")
        return value

    def _parse_expression(self, tokens, index, relationships):
        value, index = self._parse_term(tokens, index, relationships)
        while index < len(tokens) and tokens[index] == "||":
            index += 1  # Skip '||'
            right_value, index = self._parse_term(tokens, index, relationships)
            value = value or right_value
        return value, index

    def _parse_term(self, tokens, index, relationships):
        value, index = self._parse_factor(tokens, index, relationships)
        while index < len(tokens) and tokens[index] == "&&":
            index += 1  # Skip '&&'
            right_value, index = self._parse_factor(tokens, index, relationships)
            value = value and right_value
        return value, index

    def _parse_factor(self, tokens, index, relationships):
        if tokens[index] == "(":
            index += 1  # Skip '('
            value, index = self._parse_expression(tokens, index, relationships)
            if index >= len(tokens) or tokens[index] != ")":
                raise SyntaxError("Expected ')'")
            index += 1  # Skip ')'
            return value, index
        else:
            value = self._check(tokens[index], relationships)
            index += 1
            return value, index

    def _check(self, condition_str: str, relationships):
        if condition_str.startswith("between") or condition_str.startswith("aligned with"):
            anchor_1 = condition_str.split(" ")[-1]
            anchor_2 = condition_str.split(" ")[-3]
            if anchor_1 not in self.objects:
                raise UnavailableAnchorError(f"Unavailable anchor: {anchor_1}")
            if anchor_2 not in self.objects:
                raise UnavailableAnchorError(f"Unavailable anchor: {anchor_2}")
        elif condition_str.startswith("look at"):
            anchor = condition_str.split(" ")[-1]
            if anchor not in self.characters and anchor not in self.objects:
                raise UnavailableAnchorError(f"Unavailable anchor: {anchor}")
        else:
            anchor = condition_str.split(" ")[-1]
            if anchor not in self.objects:
                raise UnavailableAnchorError(f"Unavailable anchor: {anchor}")

        # if predicate is "close to", "adjacent to" is also acceptable
        if condition_str.startswith("close to"):
            anchor = condition_str.split(" ")[-1]
            return f"close to {anchor}" in relationships or f"adjacent to {anchor}" in relationships
        # if predicate is "inside", "sit on" or "lie on" is also acceptable
        if condition_str.startswith("inside"):
            anchor = condition_str.split(" ")[-1]
            return (
                f"sit on {anchor}" in relationships
                or f"lie on {anchor}" in relationships
                or f"inside {anchor}" in relationships
            )

        return condition_str in relationships
