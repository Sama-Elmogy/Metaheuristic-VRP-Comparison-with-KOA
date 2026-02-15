import numpy as np
import matplotlib.pyplot as plt
from your_koa_module import KOA, Get_Functions_detailsCEC
from benchmarks import cec14_func, cec17_func, cec20_func, cec22_func

# Parameters
N = 25  # Number of search agents (Planets)
Tmax = 200000  # Maximum number of Function evaluations
RUN_NO = 30  # Number of independent runs
Fun_id = list(range(1, 31))
cec = 3  # Change based on the desired benchmark version

# Benchmark function selection
if cec == 1:
    fhd = cec14_func
    benchmarksType = 'cec14_func'
elif cec == 2:
    fhd = cec17_func
    benchmarksType = 'cec17_func'
elif cec == 3:
    fhd = cec20_func
    benchmarksType = 'cec20_func'
elif cec == 4:
    fhd = cec22_func
    benchmarksType = 'cec22_func'
else:
    raise ValueError("Invalid CEC version selected.")

for i in range(30):
    if (cec == 3 and i >= 10) or (cec == 4 and i >= 12):
        break

    fitness = np.zeros((1, RUN_NO))

    for j in range(RUN_NO):
        if cec == 2 and i == 1:
            continue

        lb, ub, dim = Get_Functions_detailsCEC(Fun_id[i])
        fobj = Fun_id[i]

        Best_score, Best_pos, Convergence_curve = KOA(N, Tmax, ub, lb, dim, fobj, fhd)
        fitness[0, j] = Best_score

    print(f"benchmark\t{cec}\tFunction_ID\t{Fun_id[i]}\tAverage Fitness:{np.mean(fitness[0, :]):.20f}")

    # Plotting the convergence curve
    plt.figure(i + 1)
    indices = np.arange(1000, Tmax, 4000)
    plt.semilogy(Convergence_curve, '.-', markersize=6, color='red', linewidth=1.5, markevery=indices)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness obtained so far')
    plt.title(f'Function {Fun_id[i]} - KOA Convergence')
    plt.grid(True)
    plt.legend(['KOA'])
    plt.tight_layout()
    plt.show()
