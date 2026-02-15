import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist

# Load Solomon dataset
def load_solomon_data(file_path):
    data = pd.read_csv(file_path, header=None, names=["Customer", "XCOORD.", "YCOORD."])
    # Convert coordinates to numeric, handling errors
    data['XCOORD.'] = pd.to_numeric(data['XCOORD.'], errors='coerce')
    data['YCOORD.'] = pd.to_numeric(data['YCOORD.'], errors='coerce')
    return data

# Build distance matrix
def build_distance_matrix(coords):
    return cdist(coords, coords, metric='euclidean')

# Decode the solution
def decode_solution(solution):
    return [int(i) for i in solution]

# Fitness function (e.g., objective function for VRP)
def vrp_target_function(sol, customers, dist_matrix, vehicle_capacity):
    # Implement the target function logic
    # This can be based on minimizing distance, considering vehicle capacity, etc.
    return np.random.rand()  # Placeholder

# Sort the solution
def sort_solution(solution):
    customer_indices = list(range(1, len(solution) + 1))
    sorted_indices = np.argsort(solution)
    return [customer_indices[i] for i in sorted_indices]

# Plot the route
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

# Plot average convergence
def plot_average_convergence(all_convergences):
    mean_convergence = np.mean(np.array(all_convergences), axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(mean_convergence, linewidth=2, label="WA (Average)")
    plt.title("Average Convergence Curve (WA)")
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.grid()
    plt.legend()
    plt.show()

# Main Execution
if __name__ == "__main__":
    print("Running WA for Vehicle Routing Problem...")

    # Load dataset
    customers = load_solomon_data(r'C:\Users\almagd\Documents\soloman dataset\solomon_dataset\C1\C101.csv')
    coords = customers[['XCOORD.', 'YCOORD.']].values
    dist_matrix = build_distance_matrix(coords)
    vehicle_capacity = 200
    num_customers = len(customers) - 1  # excluding depot

    # Define the fitness function for optimization
    def fitness_func(sol):
        return vrp_target_function(sol, customers, dist_matrix, vehicle_capacity)

    # Optimization process
    num_runs = 10
    all_costs = []
    all_convergences = []
    best_overall_route = None
    best_overall_cost = float("inf")

    for run in range(num_runs):
        print(f"\n--- Run {run + 1} ---")
        # Replace this part with the actual WA optimizer
        # best_sol, best_cost = optimize_with_wa(fitness_func, num_customers)
        
        # For illustration, you can generate a random solution
        best_sol = np.random.rand(num_customers)  # replace with actual optimization results
        best_cost = fitness_func(best_sol)

        all_costs.append(best_cost)
        all_convergences.append(best_sol)  # Store convergence (or iterate to get best solution)

        if best_cost < best_overall_cost:
            best_overall_cost = best_cost
            best_overall_route = decode_solution(best_sol)

    # Performance Summary
    all_costs = np.array(all_costs)
    print("\n=== WA Performance Summary ===")
    print(f"Minimum Cost       : {np.min(all_costs):.2f}")
    print(f"Mean Cost          : {np.mean(all_costs):.2f}")
    print(f"Standard Deviation : {np.std(all_costs):.2f}")
    print(f"Best Route         : {best_overall_route}")
    print(f"Best Cost          : {best_overall_cost:.2f}")

    # Plots
    plot_route(best_overall_route, coords)
    plot_average_convergence(all_convergences)
