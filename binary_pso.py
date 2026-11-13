import numpy as np
from mealpy.optimizer import Optimizer

class BinaryPSO(Optimizer):
    def __init__(self, epoch=1000, pop_size=50, c1=2.0, c2=2.0, w=0.7, **kwargs):
        super().__init__(**kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        self.w = w    # inertia weight

    def initialization(self):
        self.n_dims = self.problem.n_dims
        self.pop = []
        self.velocities = []

        for _ in range(self.pop_size):
            position = np.random.randint(0, 2, self.n_dims)
            agent = self.generate_agent(position)
            self.pop.append(agent)
            self.velocities.append(np.random.rand(self.n_dims))

        self.pbest = [agent.solution.copy() for agent in self.pop]
        self.pbest_fitness = [agent.target.fitness for agent in self.pop]
        self.best = self.get_best_agent(self.pop, minmax="min")

    def evolve(self, epoch):
        for i in range(self.pop_size):
            r1 = np.random.rand(self.n_dims)
            r2 = np.random.rand(self.n_dims)

            vel = (
                self.w * self.velocities[i]
                + self.c1 * r1 * (self.pbest[i] - self.pop[i].solution)
                + self.c2 * r2 * (self.best.solution - self.pop[i].solution)
            )
            vel = np.clip(vel, -10, 10)  # optional velocity clamping
            self.velocities[i] = vel

            prob = 1 / (1 + np.exp(-vel))  # sigmoid transfer
            new_sol = np.where(np.random.rand(self.n_dims) < prob, 1, 0)

            new_agent = self.generate_agent(new_sol)
            if new_agent.target.fitness < self.pbest_fitness[i]:
                self.pbest[i] = new_sol.copy()
                self.pbest_fitness[i] = new_agent.target.fitness
            if new_agent.target.fitness < self.best.target.fitness:
                self.best = new_agent
            self.pop[i] = new_agent
