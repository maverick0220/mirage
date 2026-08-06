import networkx as nx
import numpy as np

from mirage.scoring.root_cause import rank_root_causes


if __name__ == "__main__":
    graph = nx.DiGraph()
    graph.add_edge("fuel_command", "furnace_temperature", weight=0.8, lag=1)
    graph.add_edge("furnace_temperature", "steam_pressure", weight=0.7, lag=2)
    print(
        rank_root_causes(
            np.array([2.5, 1.2, 3.0]),
            graph,
            ["fuel_command", "furnace_temperature", "steam_pressure"],
        )
    )

