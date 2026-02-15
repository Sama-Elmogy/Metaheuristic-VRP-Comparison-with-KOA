import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import os

# ---------------------------------------------
# Tabu Search for Vehicle Routing Problem (VRP)
# ---------------------------------------------

class TabuSearchVRP:
    def __init__(self, objective_func, num_customers, distance_matrix, vehicle_capacity, max_iter=1000, tabu_tenure=10):
        self.objective_func = objective_func
        self.num_customers = num_customers
        self.distance_matrix = distance_matrix
        self.vehicle_capacity = vehicle_capacity
        self.max_iter = max_iter
        self.tabu_tenure = tabu_tenure
        self.tabu_list = []
        self.best_solution = None
        self.best_cost = float('inf')
        self.current_solution = np.random.permutation(self.num_customers)
        self.current_cost = self.objective_func(self.current_solution)

    def _swap(self, solution, i, j):
        new_solution = solution.copy()
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution

    def _get_neighbors(self, solution):
        neighbors = []
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                new_solution = self._swap(solution, i, j)
                if tuple(new_solution) not in self.tabu_list:
                    neighbors.append(new_solution)
        return neighbors

    def _update_tabu_list(self, solution):
        self.tabu_list.append(tuple(solution))
        if len(self.tabu_list) > self.tabu_tenure:
            self.tabu_list.pop(0)

    def optimize(self):
        for _ in range(self.max_iter):
            neighbors = self._get_neighbors(self.current_solution)
            best_neighbor = None
            best_neighbor_cost = float('inf')

            for neighbor in neighbors:
                cost = self.objective_func(neighbor)
                if cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = cost

            if best_neighbor_cost < self.best_cost:
                self.best_solution = best_neighbor
                self.best_cost = best_neighbor_cost

            self.current_solution = best_neighbor
            self.current_cost = best_neighbor_cost
            self._update_tabu_list(best_neighbor)

        return self.best_solution, self.best_cost


# ---------------------------------------------
# VRP Functions
# ---------------------------------------------

def load_solomon_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist!")
    df = pd.read_csv(file_path)
    expected_columns = ['XCOORD.', 'YCOORD.', 'DEMAND', 'READY TIME', 'DUE DATE', 'SERVICE TIME']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError("Missing expected columns.")
    df = df[expected_columns]
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    return df

def build_distance_matrix(coords):
    return distance_matrix(coords, coords)

def vrp_target_function(solution, customers, distance_matrix, vehicle_capacity):
    sequence = decode_solution(solution)
    total_distance = 0
    load = 0
    current_time = 0
    last_customer = 0

    for customer in sequence:
        row = customers.iloc[customer]
        demand = row['DEMAND']
        ready = row['READY TIME']
        due = row['DUE DATE']
        service = row['SERVICE TIME']

        if load + demand > vehicle_capacity:
            total_distance += distance_matrix[last_customer][0]
            load = 0
            current_time = 0
            last_customer = 0

        travel_time = distance_matrix[last_customer][customer]
        arrival_time = current_time + travel_time
        if arrival_time < ready:
            arrival_time = ready
        if arrival_time > due:
            total_distance += 10000  # Penalty for lateness
        load += demand
        current_time = arrival_time + service
        last_customer = customer

    total_distance += distance_matrix[last_customer][0]
    return total_distance

def decode_solution(solution):
    customer_indices = list(range(1, len(solution) + 1))
    sorted_indices = np.argsort(solution)
    return [customer_indices[i] for i in sorted_indices]

def plot_route(route, coords):
    plt.figure(figsize=(10, 8))
    depot = coords[0]
    plt.plot(depot[0], depot[1], 'rs', markersize=10, label='Depot')

    prev = 0
    for customer in route:
        plt.plot([coords[prev][0], coords[customer][0]], [coords[prev][1], coords[customer][1]], 'bo-')
        prev = customer
    plt.plot([coords[prev][0], depot[0]], [coords[prev][1], depot[1]], 'go-')
    plt.title("Best Route Found")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()


# ---------------------------------------------
# Main Execution
# ---------------------------------------------

if __name__ == "__main__":
    print("Running Tabu Search for Vehicle Routing Problem...")

    file_path = r"C:\Users\almagd\Documents\soloman dataset\solomon_dataset\C1\C101.csv"
    customers = load_solomon_data(file_path)
    coords = customers[['XCOORD.', 'YCOORD.']].values
    dist_matrix = build_distance_matrix(coords)
    vehicle_capacity = 200
    num_customers = len(customers) - 1

    def fitness_func(sol):
        return vrp_target_function(sol, customers, dist_matrix, vehicle_capacity)

    optimizer = TabuSearchVRP(fitness_func, num_customers, dist_matrix, vehicle_capacity, max_iter=1000, tabu_tenure=20)
    best_sol, best_cost = optimizer.optimize()

    print(f"Best Cost Found: {best_cost}")
    print(f"Best Route: {decode_solution(best_sol)}")

    plot_route(decode_solution(best_sol), coords)
