from pathlib import Path
from typing import Any

import networkx as nx
import tsplib95
from pydantic import BaseModel, Field, PrivateAttr, field_validator

# Class representing the environment of the ant colony
"""
    rho: pheromone evaporation rate
"""
class Environment(BaseModel):

    rho: float = Field(..., gt=0, lt=1, description="Pheromone evaporation rate (0 < rho < 1)")
    problem_path: Path = Field(..., description="Path to .tsp instance file")

    # Private attributes
    _problem: tsplib95.models.StandardProblem = PrivateAttr()
    _num_nodes: int = PrivateAttr()
    _graph: nx.Graph = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True  # allow tsplib95 and networkx types

    @field_validator('problem_path')
    def must_be_file(cls, v: Path) -> Path:
        if not v.is_file():
            raise ValueError(f"File not found at: {v}")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self._problem = tsplib95.load(str(self.problem_path))
        self._num_nodes = self._problem.dimension
        self._build_graph()

    def _build_graph(self) -> None:
        """
        Constructs a complete undirected graph with:
          - node labels 1..n
          - edge attribute 'weight' = pseudo-Euclidean distance (ATT)
          - edge attribute 'pheromone' initialized to 0 (later set)
        """
        G = nx.Graph()
        # add nodes
        for i in range(1, self._num_nodes + 1):
            # store coordinates for potential use
            coord = self._problem.node_coords[i]
            G.add_node(i, coord=coord)
        # add edges
        for i in range(1, self._num_nodes + 1):
            for j in range(i+1, self._num_nodes + 1):
                d = self._problem.get_weight(i, j)
                G.add_edge(i, j, weight=d, pheromone=0.0)
        self._graph = G

    def initialize_pheromone_map(self, num_ants: int, start_node: int = 1) -> float:
        """
        Initialize pheromone trails on all edges: tau0 = num_ants / C_nn,
        where C_nn is the nearest-neighbor tour length.

        Returns:
            tau0: initial pheromone level.
        """
        C_nn = self.nearest_neighbor_heuristic(start_node)
        tau0 = num_ants / C_nn
        # set initial pheromone on each edge
        for u, v in self._graph.edges():
            self._graph[u][v]['pheromone'] = tau0
        return tau0

    def get_pheromone(self, i: int, j: int) -> dict[str, Any]:
        """Return pheromone level on edge (i,j)."""
        return self._graph[i][j]['pheromone']

    def get_edge_weight(self, i: int, j: int) -> dict[str, Any]:
        """Return distance weight on edge (i,j)."""
        return self._graph[i][j]['weight']

    def get_possible_locations(self) -> list[int]:
        """Return list of city labels."""
        return list(self._graph.nodes)

    def get_neighbors(self, node: int) -> list[int]:
        """Return list of neighboring city labels."""
        return list(self._graph.neighbors(node))

    def nearest_neighbor_heuristic(self, start_node: int = 1) -> float:
        """
        Compute NN heuristic tour length using graph edge weights.
        """
        unvisited = set(self._graph.nodes)
        current = start_node
        unvisited.remove(current)
        length = 0.0
        while unvisited:
            next_node = min(unvisited, key=lambda j: self.get_edge_weight(current, j))
            length += self.get_edge_weight(current, next_node)
            current = next_node
            unvisited.remove(current)
        length += self.get_edge_weight(current, start_node)
        return length

    def pheromone_evaporation(self) -> None:
        for u, v, data in self._graph.edges(data=True):
            data['pheromone'] *= (1.0 - self.rho)

    def pheromone_addition(self, paths: list[tuple[list[int], float]], Q: float = 1.0) -> None:
        for path, length in paths:
            deposit = Q / length
            for u, v in zip(path, path[1:]):
                self._graph[u][v]['pheromone'] += deposit


if __name__ == '__main__':
    env = Environment(rho=0.1, problem_path=Path('./att48-specs/att48.tsp'))
    tau0 = env.initialize_pheromone_map(num_ants=10)
    print(f'Initialized tau0 = {tau0:.6f}')
    print(f'Number of nodes: {len(env.get_possible_locations())}')
    # sample pheromone and distance on edge (1,2)
    print('Edge (1,2) weight, pheromone:',
          env.get_edge_weight(1,2), env.get_pheromone(1,2))

    
