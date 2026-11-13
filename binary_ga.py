import numpy as np
from mealpy.optimizer import Optimizer

class BinaryGA(Optimizer):
    def __init__(self, epoch=1000, pop_size=50, pc=0.9, pm=0.05, **kwargs):
        super().__init__(**kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.pc = pc  # crossover probability
        self.pm = pm  # mutation probability

    def initialization(self):
        self.n_dims = self.problem.n_dims
        self.pop = [self.generate_agent(np.random.randint(0, 2, self.n_dims)) for _ in range(self.pop_size)]
        self.best = self.get_best_agent(self.pop, minmax="min")

    def evolve(self, epoch):
        new_pop = []

        while len(new_pop) < self.pop_size:
            # --- Selection (tournament)
            p1 = self._tournament_selection()
            p2 = self._tournament_selection()

            # --- Crossover
            if np.random.rand() < self.pc:
                c1, c2 = self._uniform_crossover(p1.solution, p2.solution)
            else:
                c1, c2 = p1.solution.copy(), p2.solution.copy()

            # --- Mutation
            c1 = self._bit_flip_mutation(c1)
            c2 = self._bit_flip_mutation(c2)

            # --- Create agents
            a1 = self.generate_agent(c1)
            a2 = self.generate_agent(c2)

            new_pop.extend([a1, a2])

        # Keep best pop_size individuals
        self.pop = sorted(new_pop, key=lambda a: a.target.fitness)[:self.pop_size]
        self.best = self.get_best_agent(self.pop, minmax="min")

    def _tournament_selection(self, k=3):
        """Tournament selection with k competitors."""
        candidates = np.random.choice(self.pop, size=k, replace=False)
        return min(candidates, key=lambda a: a.target.fitness)

    def _uniform_crossover(self, p1, p2):
        """Uniform crossover operator."""
        mask = np.random.rand(self.n_dims) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2

    def _bit_flip_mutation(self, solution):
        """Bit-flip mutation operator."""
        mutation_mask = np.random.rand(self.n_dims) < self.pm
        mutated = solution.copy()
        mutated[mutation_mask] = 1 - mutated[mutation_mask]
        return mutated
