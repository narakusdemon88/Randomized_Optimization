"""
Four Peaks Implementation for Assignment 2
By: Jon-Erik Akashi (jakashi3@gatech.edu)
Date: 2/23/2023
Note: 909 is a reference to the song "One After 909"
mlrose library source: https://pypi.org/project/mlrose-hiive/
"""
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
        plt.savefig(f"../figures/{title}.png")
    plt.show()
    plt.clf()


class ROParent:
    def __init__(self, fitness):
        self.fitness = fitness
        self.restart = 50
        self.m_iter = 2_000
        self.maximum_attempt = 500

        # lists for plotting
        self.time_list = []
        self.f_evaluations = []
        self.fitnesses = []

    def process(self, size, seed=909):
        """
        Use the technique as necessary
        :param size: the current iteration
        :param seed: a random seed. default is 909
        :return: n/a
        """
        problem = mlh.DiscreteOpt(
            length=size,
            fitness_fn=self.fitness
        )

        problem.set_mimic_fast_mode(True)

        t1 = time.perf_counter()
        _, fitness, curve = mlh.random_hill_climb(problem,
                                                  max_iters=self.m_iter,
                                                  max_attempts=self.maximum_attempt,
                                                  curve=True,
                                                  restarts=self.restart,
                                                  random_state=seed)
        t2 = time.perf_counter()

        f_evaluation = curve[-1][1]
        self.f_evaluations.append(f_evaluation)
        self.fitnesses.append(fitness)
        self.time_list.append(t2-t1)


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
                                      pop_size=round(p_len/2),
                                      keep_pct=.2)
        t2 = time.perf_counter()

        f_evaluation = curve[-1][1]
        self.f_evaluations.append(f_evaluation)
        self.fitnesses.append(fitness)
        self.time_list.append(t2-t1)


class GeneticAlgorithm(ROParent):
    def __init__(self, fitness):
        super().__init__(fitness=fitness)

    def process(self, p_len, seed=909):
        population_size = round(p_len * 0.5),
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
                                            pop_size=population_size)
        t2 = time.perf_counter()

        f_evaluation = curve[-1][1]
        self.f_evaluations.append(f_evaluation)
        self.fitnesses.append(fitness)
        self.time_list.append(t2-t1)


class SimulatedAnnealing(ROParent):
    def __init__(self, fitness):
        super().__init__(fitness=fitness)
        self.schedule = mlh.ExpDecay(init_temp=2000, exp_const=.05)

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
        self.f_evaluations.append(f_evaluation)
        self.fitnesses.append(fitness)
        self.time_list.append(t2-t1)


def main():
    # Instantiate all techniques
    f = mlh.FourPeaks()
    rc = RandomizedHillClimbing(fitness=f)
    sa = SimulatedAnnealing(fitness=f)
    ga = GeneticAlgorithm(fitness=f)
    mm = MIMIC(fitness=f)

    # Create a size range to iterate through
    sizes = range(10, 301, 10)

    # Loop through the sizes
    for size in sizes:
        rc.process(size, 909)
        sa.process(size, 909)
        ga.process(size, 909)
        mm.process(size, 909)

    graphs = [
        {
            "title": "Fitness Score",
            "y_label": "Fitness Score",
            "rc_list": rc.fitnesses,
            "sa_list": sa.fitnesses,
            "ga_list": ga.fitnesses,
            "mm_list": mm.fitnesses
        },
        {
            "title": "Wall Clock Times",
            "y_label": "Time (Seconds)",
            "rc_list": rc.time_list,
            "sa_list": sa.time_list,
            "ga_list": ga.time_list,
            "mm_list": mm.time_list
        },
        {
            "title": "Function Evaluations",
            "y_label": "Total Function Evaluations #",
            "rc_list": rc.f_evaluations,
            "sa_list": sa.f_evaluations,
            "ga_list": ga.f_evaluations,
            "mm_list": mm.f_evaluations
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
            sizes=sizes,
            save_results=True),


if __name__ == "__main__":
    main()
