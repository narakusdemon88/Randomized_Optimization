"""
Neural Network Implementation Using Randomized Optimization
By: Jon-Erik Akashi (jakashi3@gatech.edu)
Date: 2/26/2023
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, learning_curve
import mlrose_hiive
import time
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np


def plot_results(iteration, lists, title, dataset):
    if ".csv" in dataset:
        dataset = dataset.replace(".csv", "")
    for i, _ in enumerate(lists):
        plt.plot(iteration, lists[i][1], label=lists[i][0])

    plt.title(f"{title} By Iterations for {dataset}")
    plt.xlabel("Max Iterations #")
    plt.ylabel("F1 Score")

    plt.legend(loc="best")

    plt.tight_layout()
    plt.grid()

    plt.savefig(f"figures/{dataset.replace('.csv', '')}/F1 Scores for {dataset.replace('.csv', '')}")

    plt.show()
    plt.clf()


def compare_techniques(X_train, X_test, y_train, y_test, techniques, iterations, dataset):
    for technique in techniques:
        if technique == "gradient_descent":
            gradient_loss, gradient_tr_f1, gradient_test_f1, gradient_time_lst = predict_results(X_train, X_test, y_train, y_test, iterations, technique)
        elif technique == "simulated_annealing":
            annealing_loss, annealing_tr_f1, annealing_test_f1, annealing_time_lst = predict_results(X_train, X_test, y_train, y_test, iterations, technique)
        elif technique == "random_hill_climb":
            hill_climb_loss, hill_climb_tr_f1, hill_climb_test_f1, hill_climb_time_lst = predict_results(X_train, X_test, y_train, y_test, iterations, technique)
        else:
            genetic_loss, genetic_tr_f1, genetic_test_f1, genetic_time_lst = predict_results(X_train, X_test, y_train, y_test, iterations, technique)

    dict_list = {
        "Train F1 Scores": [
            ("Gradient Descent", gradient_tr_f1),
            ("Randomized Hill Climbing", hill_climb_tr_f1),
            ("Simulated Annealing", annealing_tr_f1),
            ("Genetic Algorithms", genetic_tr_f1)
        ],
        "Test F1 Scores": [
            ("Gradient Descent", gradient_test_f1),
            ("Randomized Hill Climbing", hill_climb_test_f1),
            ("Simulated Annealing", annealing_test_f1),
            ("Genetic Algorithms", genetic_test_f1)]
        ,
        "Train Times": [
            ("Gradient Descent", gradient_time_lst),
            ("Randomized Hill Climbing", hill_climb_time_lst),
            ("Simulated Annealing", annealing_time_lst),
            ("Genetic Algorithms", genetic_time_lst)
        ],
        "Loss Curve": [
            ("Gradient Descent", gradient_loss),
            ("Randomized Hill Climbing", hill_climb_loss),
            ("Simulated Annealing", annealing_loss),
            ("Genetic Algorithms", genetic_loss)
        ]

    }
    for tech in dict_list:
        plot_results(iteration=iterations, lists=dict_list[tech], title=tech, dataset=dataset)


def predict_results(X_train, X_test, y_train, y_test, iteration, technique):
    hidden_nodes = [5]
    activation_func = "identity"
    early_stopping = True
    clip_max = 5
    max_attempts = 50
    random_state = 909
    schedule = mlrose_hiive.ExpDecay(exp_const=.95, init_temp=2000)
    loss = []
    tr_f1 = []
    ts_f1 = []
    time_lst = []
    for iteration in iteration:
        t1 = time.perf_counter()
        if technique == "gradient_descent":
            curr_technique = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, algorithm="gradient_descent", activation=activation_func, max_iters=iteration, learning_rate=.000015, early_stopping=early_stopping, clip_max=clip_max, max_attempts=max_attempts, random_state=random_state)
        elif technique == "random_hill_climb":
            curr_technique = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, algorithm="random_hill_climb", activation=activation_func, max_iters=iteration, learning_rate=.3, early_stopping=early_stopping, clip_max=clip_max, restarts=20, max_attempts=max_attempts, random_state=random_state)
        elif technique == "simulated_annealing":
            curr_technique = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, algorithm="simulated_annealing", activation=activation_func, schedule=schedule, max_iters=iteration, learning_rate=.3, early_stopping=early_stopping, clip_max=clip_max, max_attempts=max_attempts, random_state=random_state)
        else:
            curr_technique = mlrose_hiive.NeuralNetwork(hidden_nodes=hidden_nodes, algorithm="genetic_alg", activation=activation_func, max_iters=iteration, learning_rate=.001, early_stopping=early_stopping, clip_max=clip_max, max_attempts=max_attempts, random_state=random_state, pop_size=20, mutation_prob=.01)
        curr_technique.fit(X_train, y_train)
        t2 = time.perf_counter()

        print(iteration)

        loss.append(curr_technique.loss)
        time_lst.append(t2 - t1)
        ts_f1.append(f1_score(y_test, curr_technique.predict(X_test), average="weighted"))
        tr_f1.append(f1_score(y_train, curr_technique.predict(X_train), average="weighted"))

    return loss, tr_f1, ts_f1, time_lst


def plot_learning_curves(percentage, train_scores, cv_scores, dataset, technique):
    plt.plot(percentage, train_scores.mean(axis=1), label="Training Score")
    plt.plot(percentage, cv_scores.mean(axis=1), label="CV Score")
    plt.title(f"Learning Curve for {dataset}")
    plt.xlabel("Sample Size Percentage")
    plt.ylabel("F1 Score")

    plt.legend()
    plt.tight_layout()

    plt.grid()
    plt.savefig(f"figures/{dataset.replace('.csv', '')}/Learning Curve for {dataset} {technique}")
    plt.show()

    plt.clf()


def calc_learning_curves(X_train, y_train, dataset):
    X_train_size = len(X_train)
    percentage = [
        x / np.linspace(X_train_size / 10, X_train_size,
                        10,
                        dtype=int)[0:-1][-1] for x in np.linspace(X_train_size / 10, X_train_size, 10, dtype=int)[0:-1]
    ]

    techniques = [
        "gradient_descent",
        "random_hill_climb",
        "simulated_annealing",
        "genetic_algo"
    ]
    dataset = dataset.replace(".csv", "")
    h_nodes = [5]
    activation_technique = "identity"
    max_iterations = 1_000
    max_attempts = 50
    random_state = 909
    clip_max = 5
    early_stopping = True
    schedule = mlrose_hiive.ExpDecay(exp_const=.95, init_temp=2_000)
    train_sizes = np.linspace(len(X_train)/10, len(X_train), 10, dtype=int)[0:-1]

    for technique in techniques:
        print(technique)
        if technique == "gradient_descent" and dataset == "titanic":
            curr_technique = mlrose_hiive.NeuralNetwork(algorithm=technique, hidden_nodes=h_nodes,
                                                        max_iters=max_iterations, learning_rate=.000015,
                                                        activation=activation_technique,
                                                        clip_max=clip_max, early_stopping=early_stopping,
                                                        max_attempts=max_attempts, random_state=random_state)
        if technique == "random_hill_climb":
            curr_technique = mlrose_hiive.NeuralNetwork(algorithm=technique, hidden_nodes=h_nodes,
                                                        max_iters=max_iterations, learning_rate=.3,
                                                        activation=activation_technique,
                                                        clip_max=clip_max, restarts=20, early_stopping=early_stopping,
                                                        max_attempts=max_attempts, random_state=random_state)
        elif technique == "simulated_annealing":
            curr_technique = mlrose_hiive.NeuralNetwork(algorithm=technique, hidden_nodes=h_nodes,
                                                        schedule=schedule,
                                                        activation=activation_technique,
                                                        max_iters=max_iterations, learning_rate=.3,
                                                        clip_max=clip_max, early_stopping=early_stopping,
                                                        max_attempts=max_attempts, random_state=random_state)
        else:
            curr_technique = mlrose_hiive.NeuralNetwork(algorithm=technique, hidden_nodes=h_nodes,
                                                        max_iters=max_iterations, learning_rate=.001,
                                                        activation=activation_technique,
                                                        clip_max=clip_max, early_stopping=early_stopping,
                                                        max_attempts=max_attempts, random_state=random_state,
                                                        mutation_prob=.01, pop_size=20)

        train_sizes, train_scores, cv_scores = learning_curve(estimator=curr_technique, X=X_train, y=y_train, train_sizes=train_sizes, scoring="f1_weighted", cv=10)

        plot_learning_curves(percentage, train_scores, cv_scores, dataset, technique)


def plot_evaluations(X_train, y_train, dataset):
    if ".csv" in dataset:
        dataset = dataset.replace(".csv", "")

    activation_func = "identity"
    max_iterations = 100
    early_stopping = True
    clip_max = 5
    random_state = 909
    max_attempts = 50
    h_nodes = [5]
    schedule = mlrose_hiive.ExpDecay(exp_const=.95, init_temp=2_000)

    techniques = {
        # "gradient_descent": mlrose_hiive.NeuralNetwork(hidden_nodes=h_nodes, algorithm="gradient_descent", activation=activation_func, max_iters=max_iterations, learning_rate=.00015, early_stopping=early_stopping, clip_max=clip_max, max_attempts=max_attempts, random_state=random_state, curve=True),
        "random_hill_climb": mlrose_hiive.NeuralNetwork(hidden_nodes=h_nodes, algorithm="random_hill_climb", activation=activation_func, max_iters=max_iterations, learning_rate=.3, early_stopping=early_stopping, clip_max=clip_max, restarts=20, max_attempts=max_attempts, random_state=random_state, curve=True),
        "simulated_annealing": mlrose_hiive.NeuralNetwork(hidden_nodes=h_nodes, algorithm="simulated_annealing", activation=activation_func, schedule=schedule, max_iters=max_iterations, learning_rate=.3, early_stopping=early_stopping, clip_max=clip_max, max_attempts=max_attempts, random_state=random_state, curve=True),
        "genetic_algo": mlrose_hiive.NeuralNetwork(hidden_nodes=h_nodes, algorithm="genetic_alg", activation=activation_func, max_iters=max_iterations, learning_rate=.001, early_stopping=early_stopping, clip_max=clip_max, max_attempts=max_attempts, random_state=random_state, pop_size=20, mutation_prob=.01, curve=True)
    }

    graphs = ["Fitness Score", "Function Evaluation"]

    for i, graph in enumerate(graphs):
        for technique in techniques:
            lst = []
            curr_technique = techniques[technique]
            curr_technique.fit(X_train, y_train)
            if i == 0:
                for _, f_eval in curr_technique.fitness_curve:
                    lst.append(f_eval)
            else:
                for fitness, _ in curr_technique.fitness_curve:
                    lst.append(fitness)
            plt.plot(lst, label=technique)

        plt.title(f"{graph} Scores Per Iterations")
        plt.xlabel("Iteration #")
        plt.ylabel(graph)
        plt.legend(loc="best")
        plt.savefig(f"figures/{dataset.replace('.csv', '')}/{graph} for {dataset}.png")
        plt.show()
        plt.clf()


def main():
    for dataset in ["winequality-red.csv", "titanic.csv"]:
        print(f"Processing {dataset}")
        if dataset == "titanic":
            df = pd.read_csv(f"datasets/{dataset}.csv")

            # Drop unnecessary columns which I forgot to do in the previous assignment.
            df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

            # move the "Survived" column to the far right.
            cols = list(df.columns)
            cols.remove("Survived")
            df = df[cols + ["Survived"]]

            # change "male" to 0 and "female" to 1
            df["Sex"] = df["Sex"].map({
                "male": 0,
                "female": 1
            })
            df["Embarked"] = df["Embarked"].map({
                "S": 0,
                "C": 1,
                "Q": 2
            })

        else:  # dataset == "winequality-red"
            df = pd.read_csv("datasets/winequality-red.csv")

        # Copied from my Assignment 1 (jakashi3@gatech.edu)
        if dataset == "titanic":
            predict_col = "Survived"
            df[["Age", "Embarked"]] = SimpleImputer(strategy="mean").fit_transform(df[["Age", "Embarked"]])
        else:
            predict_col = "quality"

        X = df.drop(columns=[predict_col])
        y = df[predict_col]

        X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(df.values))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True, stratify=df.iloc[:, -1])

        # ohe = preprocessing.OneHotEncoder()
        # y_train = preprocessing.OneHotEncoder().fit_transform(y_train.values.reshape(-1, 1)).todense()
        # y_test = preprocessing.OneHotEncoder().transform(y_test.values.reshape(-1, 1)).todense()
        ohe = preprocessing.OneHotEncoder()
        y_train = ohe.fit_transform(y_train.values.reshape(-1, 1))
        y_test = ohe.transform(y_test.values.reshape(-1, 1))
        y_train = y_train.todense()
        y_test = y_test.todense()

        iterations = [i for i in range(50, 501, 50)]  # maybe change this to a larger number to see the convergence

        techniques = ["gradient_descent", "random_hill_climb", "simulated_annealing", "genetic_algo"]

        # call the main parts of the assignment

        compare_techniques(X_train, X_test, y_train, y_test, techniques, iterations, dataset=dataset)
        calc_learning_curves(X_train, y_train, dataset)
        plot_evaluations(X_train, y_train, dataset)


if __name__ == "__main__":
    main()
