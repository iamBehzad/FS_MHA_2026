import numpy as np
from mealpy.optimizer import Optimizer

class BinaryDE(Optimizer):
    def __init__(self, epoch=1000, pop_size=30, F=0.5, CR=0.9, **kwargs):
        super().__init__(**kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.F = F
        self.CR = CR

    def initialization(self):
        self.n_dims = self.problem.n_dims
        self.pop = [self.generate_agent(np.random.randint(0, 2, size=self.n_dims)) for _ in range(self.pop_size)]
        self.best = self.get_best_agent(self.pop, minmax="min")

    def evolve(self, epoch):
        new_pop = []

        for i in range(self.pop_size):
            idxs = list(range(self.pop_size))
            idxs.remove(i)
            r = lambda n: np.random.choice(idxs, n, replace=False)
            r1, r2, r3 = r(3)
            a, b, c = self.pop[r1], self.pop[r2], self.pop[r3]

            mutant = np.copy(self.pop[i].solution)
            for j in range(self.n_dims):
                if np.random.rand() < self.CR:
                    vj = a.solution[j] ^ (b.solution[j] ^ c.solution[j])  # XOR operator
                    mutant[j] = vj

            child = self.generate_agent(mutant)
            better = self.get_better_agent(self.pop[i], child, minmax="min")
            new_pop.append(better)

        self.pop = new_pop
        self.best = self.get_best_agent(self.pop, minmax="min")
 