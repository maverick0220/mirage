import numpy as np

from mirage.priors import compile_mechanism_prior, corrupt_prior, prior_conflict_score
from mirage.schemas import VariableRole, VariableSpec


def test_role_prior_and_corruption_are_deterministic():
    variables = [
        VariableSpec("context", VariableRole.CONTEXT),
        VariableSpec("command", VariableRole.ACTUATOR_COMMAND),
        VariableSpec("output", VariableRole.OUTPUT),
    ]
    prior = compile_mechanism_prior(variables, [("command", "output", 1, 0.8)])
    assert prior.allowed_mask[1, 0] == 0
    assert prior.expected_mask[1, 2] == 1
    left = corrupt_prior(prior, 0.2, seed=3)
    right = corrupt_prior(prior, 0.2, seed=3)
    np.testing.assert_array_equal(left.expected_mask, right.expected_mask)
    adjacency = np.zeros((3, 3))
    adjacency[1, 0] = 1
    assert prior_conflict_score(adjacency, prior) > 0

