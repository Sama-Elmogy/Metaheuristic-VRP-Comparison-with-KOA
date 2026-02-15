import numpy as np
import pandas as pd
import math
import os
import random
import matplotlib.pyplot as plt

# ----- Step 1: Marine Predators Algorithm (MPA) -----
def MPA(fobj, dim, SearchAgents_no, Max_iter, lb, ub):
    Top_predator = np.zeros(dim)
    Top_predator_fit = float('inf')

    Prey = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    Fitness = np.full(SearchAgents_no, np.inf)
    stepsize = np.zeros((SearchAgents_no, dim))

    cost_history = []

    for it in range(Max_iter):
        for i in range(SearchAgents_no):
            Fitness[i] = fobj(Prey[i, :])
            if Fitness[i] < Top_predator_fit:
                Top_predator_fit = Fitness[i]
                Top_predator = Prey[i, :].copy()

        CF = (1 - it / Max_iter) ** (2 * it / Max_iter)
        RL = 0.05 * levy(dim)
        RB = np.random.randn(SearchAgents_no, dim)

        for i in range(SearchAgents_no):
            if it < Max_iter / 3:  # Phase 1
                stepsize[i, :] = np.random.randn(dim) * (Top_predator - np.random.randn(dim) * Prey[i, :])
                Prey[i, :] = Prey[i, :] + np.random.rand() * stepsize[i, :]

            elif it > Max_iter * 2 / 3:  # Phase 3
                stepsize[i, :] = np.random.randn(dim) * (np.random.randn(dim) * Top_predator - Prey[i, :])
                Prey[i, :] = Top_predator + np.random.rand() * stepsize[i, :]

            else:  # Phase 2
                stepsize[i, :] = np.random.randn(dim) * (np.random.randn(dim) * Top_predator - Prey[i, :])
                Prey[i, :] = (Top_predator - Prey[i, :]) * CF + np.random.rand() * stepsize[i, :]

            if np.random.rand() < 0.2:  # Relocation
                Prey[i, :] = np.random.rand(dim) * (ub - lb) + lb

            if it > Max_iter * 2 / 3:  # Levy flight
                Prey[i, :] = Prey[i, :] + RL[i % len(RL)] * Prey[i, :]

        Prey = np.clip(Prey, lb, ub)

        cost_history.append(Top_predator_fit)
        print(f"Iteration {it + 1}: Best Cost = {Top_predator_fit:.2f}")

    return Top_predator, Top_predator_fit, cost_history

def levy(d):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    step = u / np.abs(v) ** (1 / beta)
    return step

# ----- Step 2: Read Solomon Dataset -----
def read_solomon_files(folder_path):
    customers = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            print(f"Reading {file_path}...")

            try:
                data = pd.read_csv(file_path, header=None)
                for i, row in data.iterrows():
                    try:
                        cust_id = int(row[0])
                        x = float(row[1])
                        y = float(row[2])
                        demand = float(row[3])
                        ready_time = float(row[4])
                        due_time = float(row[5])
                        service_time = float(row[6])

                        customers.append({
                            "id": cust_id,
                            "x": x,
                            "y": y,
                            "demand": demand,
                            "ready_time": ready_time,
                            "due_time": due_time,
                            "service_time": service_time
                        })
                    except ValueError as e:
                        print(f"Skipping row {i} in file {filename} due to error: {e}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    return customers

# ----- Step 3: Distance Matrix -----
def calculate_distance_matrix(customers):
    n = len(customers)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = customers[i]['x'] - customers[j]['x']
            dy = customers[i]['y'] - customers[j]['y']
            distance_matrix[i, j] = np.hypot(dx, dy)
    return distance_matrix

# ----- Step 4: Fitness Function (Route Distance) -----
def fitness_function(solution, distance_matrix):
    route = np.argsort(solution)
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i], route[i+1]]
    total_distance += distance_matrix[route[-1], route[0]]  # Return to depot
    return total_distance

# ----- Plot Route -----
def plot_route(route, customers):
    plt.figure(figsize=(10, 8))
    depot = customers[0]
    depot_coord = (depot['x'], depot['y'])
    coords = [(c['x'], c['y']) for c in customers]

    plt.plot(depot_coord[0], depot_coord[1], 'rs', markersize=10, label='Depot')

    prev = 0  # Start from depot
    for customer_idx in route:
        plt.plot([coords[prev][0], coords[customer_idx][0]],
                 [coords[prev][1], coords[customer_idx][1]], 'bo-')
        prev = customer_idx
    plt.plot([coords[prev][0], depot_coord[0]], [coords[prev][1], depot_coord[1]], 'go-')

    plt.title("Best Route Found")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()

# ----- Plot Cost History -----
def plot_cost_history(cost_history):
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, 'b-', linewidth=2)
    plt.title("Cost vs Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.grid()
    plt.show()

# ----- Main Program -----
if __name__ == "__main__":
    # ✅ Your specific dataset path
    folder_path = "C:\\Users\\almagd\\Documents\\soloman dataset\\solomon_dataset\\C1"
    
    customers = read_solomon_files(folder_path)
    distance_matrix = calculate_distance_matrix(customers)

    fobj = lambda sol: fitness_function(sol, distance_matrix)
    dim = len(customers)
    SearchAgents_no = 30
    Max_iter = 500
    lb = 0
    ub = 1

    best_solution, best_distance, cost_history = MPA(fobj, dim, SearchAgents_no, Max_iter, lb, ub)
    best_route = list(np.argsort(best_solution))

    print("\nBest Route Found:", best_route)
    print("Best Cost:", best_distance)

    plot_route(best_route, customers)
    plot_cost_history(cost_history)
