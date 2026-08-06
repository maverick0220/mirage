import numpy as np

from mirage.schemas import DynamicCausalGraph, EventRecord, VariableRole, VariableSpec


def test_variable_and_event_round_trip():
    variable = VariableSpec("pressure", VariableRole.PROCESS, unit="MPa")
    assert VariableSpec.from_dict(variable.to_dict()) == variable
    event = EventRecord("e1", 3, 8, "drift", "pressure", affected_variables=("pressure",))
    assert EventRecord.from_dict(event.to_dict()) == event


def test_dynamic_graph_networkx_orientation(tmp_path):
    weights = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
    weights[0, 0, 1, 0, 1] = 0.7
    graph = DynamicCausalGraph(weights, ("source", "target"), ("effective",))
    nx_graph = graph.to_networkx(0)
    assert nx_graph.has_edge("source", "target")
    path = graph.save(tmp_path / "graph.npz")
    restored = DynamicCausalGraph.load(path)
    np.testing.assert_allclose(restored.weights, weights)

