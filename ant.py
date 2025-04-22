from pathlib import Path
from typing import List, Tuple, Optional, Any
import random
from pydantic import BaseModel, Field, PrivateAttr
from environment import Environment


"""
    alpha: a parameter controlling the influence of the amount of pheromone during ants' path selection process
    beta: a parameter controlling the influence of the distance to the next node during ants' path selection process
"""
class Ant(BaseModel):
    alpha: float = Field(..., gt=0, description="Pheromone influence exponent")
    beta: float = Field(..., gt=0, description="Heuristic influence exponent")
    initial_location: int = Field(..., gt=0, description="Starting city label (1-indexed)")

    # Private attributes
    _environment: Optional[Environment] = PrivateAttr(default=None)
    _current_location: int = PrivateAttr()
    _traveled_distance: float = PrivateAttr(default=0.0)
    _visited: set[int] = PrivateAttr(default_factory=set)
    _path: List[int] = PrivateAttr(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def join(self, environment) -> None:
        """Place the ant in the environment and reset its tour state."""
        self._environment = environment
        self._current_location = self.initial_location
        self._traveled_distance = 0.0
        self._visited = {self._current_location}
        self._path = [self._current_location]

    def get_distance(self, i: int, j: int) -> float:
        """Fetch pseudo-Euclidean distance from the environment."""
        return self._environment.get_edge_weight(i, j)

    def select_path(self) -> int:
        """
        Random-proportional selection: p_ij ∝ τ_ij^α · η_ij^β.
        Updates state and returns next city.
        """
        env = self._environment
        curr = self._current_location
        candidates = [c for c in env.get_possible_locations() if c not in self._visited]

        # compute selection weights
        weights = []
        for j in candidates:
            tau = env.get_pheromone(curr, j)
            d = self.get_distance(curr, j)
            eta = 1.0 / d if d > 0 else 0.0
            weights.append((tau ** self.alpha) * (eta ** self.beta))

        total = sum(weights)
        if total <= 0:
            next_city = random.choice(candidates)
        else:
            r = random.random()
            cum = 0.0
            next_city = candidates[-1]
            for j, w in zip(candidates, weights):
                cum += w / total
                if r <= cum:
                    next_city = j
                    break

        # update tour
        dist = self.get_distance(curr, next_city)
        self._traveled_distance += dist
        self._current_location = next_city
        self._visited.add(next_city)
        self._path.append(next_city)
        return next_city

    def run(self, return_to_start: bool = True) -> Tuple[List[int], float]:
        """Construct a full tour and return (path, total distance)."""
        n = len(self._environment.get_possible_locations())
        while len(self._visited) < n:
            self.select_path()
        if return_to_start:
            start = self._path[0]
            dist = self.get_distance(self._current_location, start)
            self._traveled_distance += dist
            self._path.append(start)
            self._current_location = start
        return self._path, self._traveled_distance


if __name__ == "__main__":
    env = Environment(rho=0.1, problem_path=Path('./att48-specs/att48.tsp'))
    print(env.get_edge_weight(1,2))
    ant = Ant(alpha=1.0, beta=5.0, initial_location=1)
    ant.join(env)
    path, dist = ant.run(return_to_start=True)
    print(path, dist)