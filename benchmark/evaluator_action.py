import re


class UnavailableActionError(Exception):
    pass


class ActionEvaluator:
    def __init__(self):
        self.available_action_labels = []

    def reset_available_action_labels(self, available_action_labels):
        self.available_action_labels = available_action_labels.copy()
        self.available_action_labels += ["sit", "lie"]

    def evaluate(self, condition_str, target_actions):
        if not all(a in self.available_action_labels for a in target_actions):
            raise UnavailableActionError(f"Unavailable action labels: {target_actions}")

        if not condition_str:
            return True

        tokens = re.split(r"(\|\||&&|\(|\))", condition_str)
        tokens = [t.strip() for t in tokens if t.strip() != ""]
        value, index = self._parse_expression(tokens, 0, target_actions)
        if index != len(tokens):
            raise SyntaxError("Unexpected tokens at the end")
        return value

    def _parse_expression(self, tokens, index, target_actions):
        value, index = self._parse_term(tokens, index, target_actions)
        while index < len(tokens) and tokens[index] == "||":
            index += 1  # Skip '||'
            right_value, index = self._parse_term(tokens, index, target_actions)
            value = value or right_value
        return value, index

    def _parse_term(self, tokens, index, target_actions):
        value, index = self._parse_factor(tokens, index, target_actions)
        while index < len(tokens) and tokens[index] == "&&":
            index += 1  # Skip '&&'
            right_value, index = self._parse_factor(tokens, index, target_actions)
            value = value and right_value
        return value, index

    def _parse_factor(self, tokens, index, target_actions):
        if tokens[index] == "(":
            index += 1  # Skip '('
            value, index = self._parse_expression(tokens, index, target_actions)
            if index >= len(tokens) or tokens[index] != ")":
                raise SyntaxError("Expected ')'")
            index += 1  # Skip ')'
            return value, index
        else:
            value = self._check(tokens[index], target_actions)
            index += 1
            return value, index

    def _check(self, condition_str: str, target_actions):
        if condition_str.startswith("!"):
            return condition_str[1:] not in target_actions
        return condition_str in target_actions
