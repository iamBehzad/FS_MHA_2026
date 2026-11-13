import numpy as np
from collections import deque
from mealpy.optimizer import Optimizer


class ARBOS(Optimizer):
    def __init__(self, epoch=1000, pop_size=50, memory_size=20, stagnation_gen=10, elite_ratio=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.memory_size = memory_size
        self.stagnation_gen = stagnation_gen
        self.elite_ratio = elite_ratio

        self.operators = [
            "bit_flip", "xor_crossover", "swap",
            "inversion", "random_restart", "uniform_crossover"
        ]
        self.q_values = {op: 0.0 for op in self.operators}
        self.memory = {op: deque(maxlen=self.memory_size) for op in self.operators}
        self.no_improve_count = 0
        self.mutation_rate = 0.3  # adaptive

    def initialization(self):
        self.n_dims = self.problem.n_dims
        self.pop = [self.generate_agent(np.random.randint(0, 2, self.n_dims)) for _ in range(self.pop_size)]
        self.best = self.get_best_agent(self.pop, minmax="min")

    def evolve(self, epoch):
        new_pop = []
        improved = False

        elites = sorted(self.pop, key=lambda a: a.target.fitness)[:int(self.elite_ratio * self.pop_size)]

        for i in range(self.pop_size):
            op = self.select_operator(epsilon=0.1)
            parent = self.pop[i].solution
            mutated = self.apply_operator(op, parent, i, epoch)
            child = self.generate_agent(mutated)

            f_parent = self.pop[i].target.fitness
            f_child = child.target.fitness

            if f_child < f_parent:
                new_pop.append(child)
                reward = f_parent - f_child
                self.update_q(op, reward)
                if f_child < self.best.target.fitness:
                    self.best = child
                    improved = True
            else:
                new_pop.append(self.pop[i])
                self.update_q(op, 0)

        # 🔄 Adaptive mutation adjustment
        self.mutation_rate = max(0.05, self.mutation_rate * 0.95) if improved else min(0.5, self.mutation_rate * 1.05)

        # ♻️ Replace worst individuals with elites
        new_pop = sorted(new_pop, key=lambda a: a.target.fitness)
        new_pop[-len(elites):] = elites
        self.pop = new_pop

        self.no_improve_count = 0 if improved else self.no_improve_count + 1
        if self.no_improve_count >= self.stagnation_gen:
            self.diversify_population()
            self.no_improve_count = 0

    def diversify_population(self):
        print("[ARBOS] Diversify population due to stagnation...")
        for i in range(len(self.pop)):
            if np.random.rand() < 0.3:
                new_sol = np.random.randint(0, 2, size=self.n_dims)
                self.pop[i] = self.generate_agent(new_sol)

    def select_operator(self, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.choice(self.operators)
        return max(self.q_values, key=self.q_values.get)

    def update_q(self, op, reward):
        self.memory[op].append(reward)
        avg_reward = np.mean(self.memory[op]) if self.memory[op] else 0.0
        self.q_values[op] = 0.9 * self.q_values[op] + 0.1 * avg_reward

    def apply_operator(self, op, x, idx, epoch):
        x = np.copy(x)
        rate = self.mutation_rate

        if op == "bit_flip":
            mask = np.random.rand(self.n_dims) < rate
            x[mask] = 1 - x[mask]
        elif op == "xor_crossover":
            partner = self.pop[np.random.choice([j for j in range(self.pop_size) if j != idx])].solution
            mask = np.random.rand(self.n_dims) < 0.5
            x[mask] = np.bitwise_xor(x[mask], partner[mask])
        elif op == "swap":
            a, b = np.random.choice(self.n_dims, 2, replace=False)
            x[a], x[b] = x[b], x[a]
        elif op == "inversion":
            a, b = sorted(np.random.choice(self.n_dims, 2, replace=False))
            x[a:b+1] = x[a:b+1][::-1]
        elif op == "random_restart":
            mask = np.random.rand(self.n_dims) < rate
            x[mask] = np.random.randint(0, 2, size=np.sum(mask))
        elif op == "uniform_crossover":
            partner = self.pop[np.random.choice([j for j in range(self.pop_size) if j != idx])].solution
            mask = np.random.rand(self.n_dims) < 0.5
            x[mask] = partner[mask]

        return x