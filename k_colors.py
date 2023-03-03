"""
K Colors Implementation for Assignment 2
By: Jon-Erik Akashi (jakashi3@gatech.edu)
Date: 2/24/2023
Note: 909 is a reference to the song "One After 909"
mlrose library source: https://pypi.org/project/mlrose-hiive/
"""
import numpy as np
import matplotlib.pyplot as plt
import mlrose_hiive as mlh
import time


def plot_results(
        rc_list: list,
        sa_list: list,
        ga_list: list,
        mm_list: list,
        title: str,
        xlabel: str,
        ylabel: str,
        sizes,
        save_results=False):
    """
    Plot results and save them to the figure directory
    :param rc_list: list
    :param sa_list: list
    :param ga_list: list
    :param mm_list: list
    :param title: string
    :param xlabel: string
    :param ylabel: string
    :param sizes: list
    :param save_results: boolean
    :return: n/a
    """
    technique_lists = [
        (rc_list, "Randomized Hill Climbing"),
        (mm_list, "MIMIC"),
        (ga_list, "Genetic Algo"),
        (sa_list, "Simulated Annealing")
    ]
    for technique in technique_lists:
        technique_list = technique[0]
        label = technique[1]
        plt.plot(sizes, technique_list, label=label)

    # Title, X Axis, Y Axis
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Legend, optimize layout, add grid
    plt.legend()
    plt.tight_layout()
    plt.grid()

    # Toggle saving and display
    if save_results:
        plt.savefig(f"../figures/K Color/{title}.png")
    plt.show()
    plt.clf()


class ROParent:
    def __init__(self, fitness):
        self.fitness = fitness
        self.restart = 25
        self.m_iter = 200
        self.maximum_attempt = 20

        # lists for plotting
        self.time = 0
        self.f_evaluation = 0
        self.one_fitness = 0

    def process(self, p_len, seed=909):
        """
        Use the technique as necessary
        :param p_len: the current iteration
        :param seed: a random seed. default is 909
        :return: n/a
        """
        problem = mlh.DiscreteOpt(
            length=p_len,
            max_val=2,
            maximize=False,
            fitness_fn=self.fitness
        )

        problem.set_mimic_fast_mode(True)

        t1 = time.perf_counter()
        _, fitness, curve = mlh.random_hill_climb(problem,
                                                  max_attempts=self.maximum_attempt,
                                                  max_iters=self.m_iter,
                                                  restarts=self.restart,
                                                  curve=True,
                                                  random_state=seed)
        t2 = time.perf_counter()

        f_evaluation = curve[-1][1]
        self.f_evaluation = f_evaluation
        self.one_fitness = fitness
        self.time = t2 - t1


class RandomizedHillClimbing(ROParent):
    def __init__(self, fitness):
        super().__init__(fitness=fitness)


class MIMIC(ROParent):
    def __init__(self, fitness):
        super().__init__(fitness=fitness)

    def process(self, p_len, seed=909):
        print(p_len)
        problem = mlh.DiscreteOpt(
            length=p_len,
            fitness_fn=self.fitness
        )
        problem.set_mimic_fast_mode(True)

        t1 = time.perf_counter()
        _, fitness, curve = mlh.mimic(problem,
                                      curve=True,
                                      max_iters=self.m_iter,
                                      max_attempts=self.maximum_attempt,
                                      random_state=seed,
                                      pop_size=500,
                                      keep_pct=.2)
        t2 = time.perf_counter()

        f_evaluation = curve[-1][1]
        self.f_evaluation = f_evaluation
        self.one_fitness = fitness
        self.time = t2 - t1


class GeneticAlgorithm(ROParent):
    def __init__(self, fitness):
        super().__init__(fitness=fitness)

    def process(self, p_len, seed=909):
        population_size = 1200
        mutation_probability = 0.2
        print(p_len)
        problem = mlh.DiscreteOpt(
            length=p_len,
            fitness_fn=self.fitness
        )
        problem.set_mimic_fast_mode(True)

        t1 = time.perf_counter()
        _, fitness, curve = mlh.genetic_alg(problem,
                                            random_state=seed,
                                            curve=True,
                                            max_iters=self.m_iter,
                                            mutation_prob=mutation_probability,
                                            max_attempts=self.maximum_attempt,
                                            pop_size=int(population_size),
                                            )
        t2 = time.perf_counter()

        f_evaluation = curve[-1][1]
        self.f_evaluation = f_evaluation
        self.one_fitness = fitness
        self.time = t2 - t1


class SimulatedAnnealing(ROParent):
    def __init__(self, fitness):
        super().__init__(fitness=fitness)
        self.schedule = mlh.ExpDecay(init_temp=2_000, exp_const=.05)

    def process(self, p_len, seed=909):
        print(p_len)
        problem = mlh.DiscreteOpt(
            length=p_len,
            fitness_fn=self.fitness
        )
        problem.set_mimic_fast_mode(True)

        t1 = time.perf_counter()
        _, fitness, curve = mlh.simulated_annealing(problem,
                                                    random_state=seed,
                                                    curve=True,
                                                    max_iters=self.m_iter,
                                                    schedule=self.schedule,
                                                    max_attempts=self.maximum_attempt)
        t2 = time.perf_counter()

        f_evaluation = curve[-1][1]
        self.f_evaluation = f_evaluation
        self.one_fitness = fitness
        self.time = t2 - t1


def create_edges(edge_size):
    tracker = [0, 1]
    edges = [(0, 1)]
    for size in range(edge_size):
        edge = (np.random.choice(tracker), np.random.randint(0, edge_size))
        while edge in edges or edge[0] == edge[1] or (edge[1], edge[0]) in edges:
            edge = (np.random.choice(tracker), np.random.randint(0, edge_size))
        if edge[1] not in tracker:
            tracker.append(edge[1])
        edges.append(edge)
    return edges


def main():
    master_f_evals = []
    master_fitnesses = []
    master_time_list = []

    technique_list = [
        ("Randomized Hill Climbing", RandomizedHillClimbing),
        ("MIMIC", MIMIC),
        ("Genetic Algorithm", GeneticAlgorithm),
        ("Simulated Annealing", SimulatedAnnealing)
    ]

    for technique in technique_list:
        print(technique[0])
        f_evals = []
        fitnesses = []
        time_list = []

        edge_sizes = range(10, 201, 10)
        for edge_size in edge_sizes:
            print(edge_size)
            edges = create_edges(edge_size=edge_size)
            fitness = mlh.MaxKColor(edges)
            problem = mlh.DiscreteOpt(length=edge_size, fitness_fn=fitness, maximize=False, max_val=2)
            problem.set_mimic_fast_mode(True)

            curr_technique = technique[1](fitness=fitness)
            curr_technique.process(p_len=edge_size)
            f_evals.append(curr_technique.f_evaluation)
            fitnesses.append(curr_technique.one_fitness)
            time_list.append(curr_technique.time)

        master_f_evals.append((technique[0], f_evals))
        master_fitnesses.append((technique[0], fitnesses))
        master_time_list.append((technique[0], time_list))

    graphs = [
        {
            "title": "Match Color Edge #",
            "y_label": "Match Color Edge #",
            "rc_list": master_fitnesses[0][1],
            "sa_list": master_fitnesses[3][1],
            "ga_list": master_fitnesses[2][1],
            "mm_list": master_fitnesses[1][1]
        },
        {
            "title": "Wall Clock Times",
            "y_label": "Time (Seconds)",
            "rc_list": master_time_list[0][1],
            "sa_list": master_time_list[3][1],
            "ga_list": master_time_list[2][1],
            "mm_list": master_time_list[1][1]
        },
        {
            "title": "Function Evaluations",
            "y_label": "Total Function Evaluations #",
            "rc_list": master_f_evals[0][1],
            "sa_list": master_f_evals[3][1],
            "ga_list": master_f_evals[2][1],
            "mm_list": master_f_evals[1][1]
        }
    ]

    # Iterate through the list of dictionaries w/ information for plotting
    for graph in graphs:
        plot_results(
            rc_list=graph["rc_list"],
            sa_list=graph["sa_list"],
            ga_list=graph["ga_list"],
            mm_list=graph["mm_list"],
            title=f"{graph['title']} versus size",
            xlabel="Size of Samples",
            ylabel=graph["y_label"],
            sizes=edge_sizes,
            save_results=True),


if __name__ == "__main__":
    main()
