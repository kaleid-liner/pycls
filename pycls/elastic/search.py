# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random
import numpy as np
from tqdm import tqdm

import pandas as pd


__all__ = ["EvolutionFinder"]


def calculate_R(efficiency, acc):
    energy, latency = efficiency
    return (acc / 1000000000 + 2.5018) / energy ** 0.7125


class EvolutionFinder:
    def __init__(self, efficiency_predictor, accuracy_predictor, arch_encoder, **kwargs):
        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor
        self.arch_encoder = arch_encoder

        # evolution hyper-parameters
        self.arch_mutate_prob = kwargs.get("arch_mutate_prob", 0.1)
        self.resolution_mutate_prob = kwargs.get("resolution_mutate_prob", 0.5)
        self.population_size = kwargs.get("population_size", 100)
        self.max_time_budget = kwargs.get("max_time_budget", 500)
        self.parent_ratio = kwargs.get("parent_ratio", 0.25)
        self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)

    @property
    def arch_manager(self):
        return self.arch_encoder

    def update_hyper_params(self, new_param_dict):
        self.__dict__.update(new_param_dict)

    def random_valid_sample(self, constraint):
        while True:
            sample = {}
            self.arch_manager.random_sample(raw_arch=sample)
            efficiency = self.efficiency_predictor.get_efficiency(sample)
            acc = self.accuracy_predictor.get_efficiency(sample)
            return sample, efficiency, acc

    def mutate_sample(self, sample, constraint):
        while True:
            new_sample = self.arch_manager.mutate(sample, self.arch_mutate_prob)

            efficiency = self.efficiency_predictor.get_efficiency(new_sample)
            acc = self.accuracy_predictor.get_efficiency(new_sample)
            return new_sample, efficiency, acc

    def crossover_sample(self, sample1, sample2, constraint):
        while True:
            new_sample = self.arch_manager.crossover(sample1, sample2)

            efficiency = self.efficiency_predictor.get_efficiency(new_sample)
            acc = self.accuracy_predictor.get_efficiency(new_sample)
            return new_sample, efficiency, acc

    def run_evolution_search(self, constraint, verbose=False, **kwargs):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        self.update_hyper_params(kwargs)

        mutation_numbers = int(round(self.mutation_ratio * self.population_size))
        parents_size = int(round(self.parent_ratio * self.population_size))

        best_valids = [-100]
        population = []  # (R, sample, energy, flops) tuples
        child_pool = []
        efficiency_pool = []
        acc_pool = []
        candidates = []
        best_info = None
        if verbose:
            print("Generate random population...")
        for _ in range(self.population_size):
            sample, efficiency, acc = self.random_valid_sample(constraint)
            child_pool.append(sample)
            efficiency_pool.append(efficiency)
            acc_pool.append(acc)
            print(sample, efficiency)

        for i in range(self.population_size):
            print(i)
            item = (calculate_R(efficiency_pool[i], acc_pool[i]), child_pool[i], efficiency_pool[i], acc_pool[i])
            candidates.append(item)
            population.append(item)

        df = pd.DataFrame.from_dict(population)
        df.to_csv('population.csv', index=False)
        df = pd.DataFrame.from_dict(candidates)
        df.to_csv('candidates.csv', index=False)

        if verbose:
            print("Start Evolution...")
        # After the population is seeded, proceed with evolving the population.
        with tqdm(
            total=self.max_time_budget,
            desc="Searching with constraint (%s)" % constraint,
            disable=(not verbose),
        ) as t:
            for i in range(self.max_time_budget):
                parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
                R = parents[0][0]
                t.set_postfix({"R": parents[0][0]})
                if not verbose and (i + 1) % 100 == 0:
                    print("Iter: {} R: {}".format(i + 1, parents[0][0]))

                if R > best_valids[-1]:
                    best_valids.append(R)
                    best_info = parents[0]
                else:
                    best_valids.append(best_valids[-1])

                population = parents
                child_pool = []
                efficiency_pool = []
                acc_pool = []

                for j in range(mutation_numbers):
                    par_sample = population[np.random.randint(parents_size)][1]
                    # Mutate
                    new_sample, efficiency, acc = self.mutate_sample(par_sample, constraint)
                    child_pool.append(new_sample)
                    efficiency_pool.append(efficiency)
                    acc_pool.append(acc)

                for j in range(self.population_size - mutation_numbers):
                    par_sample1 = population[np.random.randint(parents_size)][1]
                    par_sample2 = population[np.random.randint(parents_size)][1]
                    # Crossover
                    new_sample, efficiency, acc = self.crossover_sample(
                        par_sample1, par_sample2, constraint
                    )
                    child_pool.append(new_sample)
                    efficiency_pool.append(efficiency)
                    acc_pool.append(acc)

                for j in range(self.population_size):
                    item = (calculate_R(efficiency_pool[j], acc_pool[j]), child_pool[j], efficiency_pool[j], acc_pool[j])
                    population.append(item)
                    candidates.append(item)

                df = pd.DataFrame.from_dict(population)
                df.to_csv('population.csv', index=False)
                df = pd.DataFrame.from_dict(candidates)
                df.to_csv('candidates.csv', index=False)

                t.update(1)

        return best_valids, best_info
