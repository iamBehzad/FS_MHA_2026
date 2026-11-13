import numpy as np
from mealpy.optimizer import Optimizer

class BinaryGWO(Optimizer):
    def __init__(self, epoch=1000, pop_size=30, **kwargs):
        super().__init__(**kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def initialization(self):
        self.n_dims = self.problem.n_dims
        self.pop = [self.generate_agent(np.random.randint(0, 2, self.n_dims)) for _ in range(self.pop_size)]
        self.alpha = self.beta = self.delta = self.get_best_agent(self.pop, minmax="min")

    def evolve(self, epoch):
        a = 2 - 2 * (epoch / self.epoch)  # linearly decreasing from 2 to 0
        new_pop = []

        # Update alpha, beta, delta
        sorted_agents = sorted(self.pop, key=lambda agent: agent.target.fitness)
        self.alpha, self.beta, self.delta = sorted_agents[:3]

        for i in range(self.pop_size):
            x = self.pop[i].solution.astype(float)
            x_new = np.zeros_like(x)

            for j in range(self.n_dims):
                r1, r2 = np.random.rand(2)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * self.alpha.solution[j] - x[j])
                X1 = self.alpha.solution[j] - A1 * D_alpha

                r1, r2 = np.random.rand(2)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * self.beta.solution[j] - x[j])
                X2 = self.beta.solution[j] - A2 * D_beta

                r1, r2 = np.random.rand(2)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * self.delta.solution[j] - x[j])
                X3 = self.delta.solution[j] - A3 * D_delta

                xj = (X1 + X2 + X3) / 3.0
                s = 1 / (1 + np.exp(-xj))  # sigmoid
                x_new[j] = 1 if np.random.rand() < s else 0

            new_agent = self.generate_agent(x_new.astype(int))
            new_pop.append(new_agent)

        self.pop = new_pop
        self.best = self.get_best_agent(self.pop, minmax="min")
