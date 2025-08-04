from src.type import Object

from .evaluator_action import ActionEvaluator
from .evaluator_position import PositionEvaluator
from .evaluator_relationships import RelationshipEvaluator


class CharacterNotFoundError(Exception):
    pass


class UnavailableActionError(Exception):
    pass


class Evaluator:
    def __init__(self):
        self.action_evaluator = ActionEvaluator()
        self.relationship_evaluator = RelationshipEvaluator()
        self.position_evaluator = PositionEvaluator()

    def setup_evaluator(self, available_action_labels: list[str], characters: list[str], objects: list[Object]):
        """
        This method sets up the evaluator with the given available action labels, characters, and objects.
        This function must be called before evaluating any test cases to ensure that the evaluator has the proper context.
        """
        self.action_evaluator.reset_available_action_labels(available_action_labels)
        self.relationship_evaluator.reset_available_anchors(characters, objects)
        self.position_evaluator.reset_objects(objects)

    def evaluate_result(self, test_case, planning_result):
        """
        This method checks if the planning result matches the expected plans in the test case.
        Before using this method, `setup_evaluator` must be called to provide the necessary context.

        The planning result to be structured as follows:
            {
                "character_name": {
                    "target_action": [list of action labels (e.g. "drink", "chat", "use_coffee_maker", etc.)],
                    "relationships": [list of relationships (e.g. "close to table_1", "sit on chair_2", etc.)],
                    "sampled_position": [2D coordinates representing the xy position of the character],
                },
                ...
            }
        In case of the test cases with PI tag, only the `sampled_position` is required and checked.
        In other cases (= test cases with OA, RC, SS tags), only the `target_action` and `relationships` are required and checked.
        If the test case allows a null plan, an empty planning result is also considered as passed.
        """

        if test_case["allow_null_plan"] and not planning_result:
            return True

        passed = []
        for expected_plan in test_case["expected_plans"]:
            try:
                if "PI" in test_case["tags"]:
                    passed_ = self._evaluate_position(expected_plan, planning_result)
                else:
                    passed_ = self._evaluate(expected_plan, planning_result)
                passed.append(passed_)
            except Exception as e:
                raise e
        return all(passed)

    def _evaluate(self, expected_plan, planning_result):
        if expected_plan["character"] not in planning_result.keys():
            raise CharacterNotFoundError(f"{expected_plan['character']} not found in planning result")

        character_plan = planning_result[expected_plan["character"]]
        passed = self.action_evaluator.evaluate(
            expected_plan["target_action"],
            character_plan["target_action"],
        )
        if not passed:
            return False

        passed = self.relationship_evaluator.evaluate(
            expected_plan["relationships"],
            character_plan["relationships"],
        )
        if not passed:
            return False
        return True

    def _evaluate_position(self, expected_plan, planning_result):
        if expected_plan["character"] not in planning_result.keys():
            raise CharacterNotFoundError(f"{expected_plan['character']} not found in parsed event")

        character_plan = planning_result[expected_plan["character"]]
        passed, error = self.position_evaluator.evaluate(
            expected_plan["sampled_position"],
            character_plan["sampled_position"],
        )
        collision_occurred, collision_error = self.position_evaluator.check_collision(
            character_plan["sampled_position"]
        )
        return passed and not collision_occurred
