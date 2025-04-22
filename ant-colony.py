import json
from pathlib import Path
from typing import List, Tuple

import random

import tsplib95
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from environment import Environment
from ant import Ant 


"""
    ant_population: the number of ants in the ant colony
    iterations: the number of iterations 
    alpha: a parameter controlling the influence of the amount of pheromone during ants' path selection process
    beta: a parameter controlling the influence of the distance to the next node during ants' path selection process
    rho: pheromone evaporation rate
"""
class AntColony(BaseModel):
    ant_population: int = Field(..., gt=0, description="Number of ants")
    iterations: int = Field(..., gt=0, description="Number of iterations")
    alpha: float = Field(..., gt=0, description="Pheromone exponent")
    beta: float = Field(..., gt=0, description="Heuristic exponent")
    rho: float = Field(..., gt=0, lt=1, description="Pheromone evaporation rate")
    problem_path: Path = Field(..., description="Path to .tsp instance file")

    # Private attributes
    _environment: Environment = PrivateAttr()
    _ants: List[Ant] = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    @field_validator('problem_path')
    def validate_path(cls, v: Path) -> Path:
        if not v.is_file():
            raise ValueError(f"File not found at: {v}")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self._environment = Environment(rho=self.rho, problem_path=self.problem_path)
        self._environment.initialize_pheromone_map(num_ants=self.ant_population)
        self._ants = []
        for _ in range(self.ant_population):
            start = random.choice(self._environment.get_possible_locations())
            ant = Ant(alpha=self.alpha, beta=self.beta, initial_location=start)
            ant.join(self._environment)
            self._ants.append(ant)

    def solve(self) -> Tuple[List[int], float, List[dict]]:
        """
        Run the Ant System algorithm, returning the best tour, its length, and all routes.
        """
        best_solution: List[int] = []
        shortest_distance: float = float('inf')
        routes: List[dict] = []

        for iteration in range(self.iterations):
            for ant in self._ants:
                # randomize start each iteration
                start = random.choice(self._environment.get_possible_locations())
                ant.initial_location = start
                ant.join(self._environment)
                tour, dist = ant.run(return_to_start=True)
                routes.append({
                    'iteration': iteration,
                    'tour': tour,
                    'distance': dist
                })
                if dist < shortest_distance:
                    shortest_distance = dist
                    best_solution = tour.copy()

            # pheromone update
            self._environment.pheromone_evaporation()
            deposits = [(ant._path, ant._traveled_distance) for ant in self._ants]
            self._environment.pheromone_addition(deposits)

        return best_solution, shortest_distance, routes

    def save_routes(self, path: Path) -> None:
        """Save collected routes to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.solve()[2], f)

def load_optimal_tour(tour_file: Path) -> List[int]:
    sol = tsplib95.load(str(tour_file))
    # TSPLIB95 marks the loaded object’s type as 'TOUR' and populates `.tours`
    tours = getattr(sol, "tours", None)
    if not tours:
        raise ValueError(f"No tours found in {tour_file}")
    return tours[0]

def main():
    alphas = [2]
    betas = [5]
    rhos = [0.3]

    for alpha in alphas:
        for beta in betas:
            for rho in rhos:
                ant_colony = AntColony(ant_population=20,
                                       iterations=50,
                                       alpha=alpha,
                                       beta=beta,
                                       rho=rho,
                                       problem_path=Path(f'./att48-specs/att48.tsp'))

                # Solve the ant colony optimization problem
                solution, distance, routes = ant_colony.solve()
                print("Solution: ", solution)
                print("Distance: ", distance)

                optimal_tour = load_optimal_tour(Path(f"./att48-specs/att48.opt.tour"))
                # print(optimal_tour)
                if solution == optimal_tour:
                    print("Found optimal tour!")
                    print(solution)

                # ant_colony.save_routes(path=Path(f'data/test_refactoring_routes_10_alpha_{alpha}_beta_{beta}_rho_{rho}.json'))



if __name__ == '__main__':
    main()