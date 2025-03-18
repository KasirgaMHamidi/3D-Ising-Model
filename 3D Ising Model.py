import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ************************************
# Generating the 3-Dimensional Jij Matrix with Periodic Boundary Conditions (Thorus)
# ************************************
def generate_J_matrix_3D(L, seed=None, J_distribution="uniform"):
    if seed is not None:
        np.random.seed(seed)
    N = L ** 3
    J = np.zeros((N, N))

    def index(x, y, z):
        return ((x % L) * (L ** 2)) + ((y % L) * L) + (z % L)

    directions = [(1, 0, 0), (-1, 0, 0),
                  (0, 1, 0), (0, -1, 0),
                  (0, 0, 1), (0, 0, -1)]

    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = index(x, y, z)
                for dx, dy, dz in directions:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    j = index(nx, ny, nz)
                    if i < j:
                        if J_distribution == "uniform":
                            value = np.random.uniform(-1, 1)
                        else:
                            value = np.random.normal(0, 1)
                        J[i, j] = value
                        J[j, i] = value  # Enforce symmetry
    return J


# ************************************
# Computing the Hamiltonians
# ************************************
def compute_hamiltonians(J, num_samples=10000):
    N = J.shape[0]
    total_states = 2 ** N

    def compute_H(spins):
        return -0.5 * np.sum(J * np.outer(spins, spins))  # 1/2 avoids double-counting

    if total_states <= num_samples:
        hamiltonians = np.empty(
            total_states)  # Exact enumaration if the system size is small, otherwise Monte Carlo Sampling
        configurations = []
        for state_int in range(total_states):
            spins = np.array([1 if (state_int >> i) & 1 else -1 for i in range(N)])
            H = compute_H(spins)
            hamiltonians[state_int] = H
            configurations.append(spins)
        scale_factor = 1.0
        return hamiltonians, configurations, scale_factor
    else:
        hamiltonians = np.empty(num_samples)
        configurations = []
        for i in range(num_samples):
            spins = np.random.choice([1, -1], size=N)
            H = compute_H(spins)
            hamiltonians[i] = H
            configurations.append(spins)
        scale_factor = total_states / num_samples
        return hamiltonians, configurations, scale_factor


# ************************************
# Function for finding the ground state of the Hamiltonian
# ************************************
def find_ground_state(hamiltonians):
    return np.min(hamiltonians)


# ************************************
# Smoothes out the Step Cumulative Function with convulution padding
# ************************************

def compute_smoothed_cumulative(hamiltonians, scale_factor, sigma=0.1, num_points=300, extra_margin=0.2):
    """
    Computes a smoothed cumulative function F(E) (number of states with energy ≤ E).
    1. Sort the sampled energies
    2. Build a step cumulative function
    3. Multiply by scale_factor
    4. Convolve with Gaussian
    """
    sorted_energies = np.sort(hamiltonians)
    E_min = sorted_energies[0]
    E_max = sorted_energies[-1]
    x_grid = np.linspace(E_min - extra_margin, E_max + extra_margin, num_points)

    F_sampled = np.array([np.searchsorted(sorted_energies, E, side='right') for E in x_grid])
    F = F_sampled * scale_factor  # scale to represent full space

    kernel_size = 100
    kernel_x = np.linspace(-3 * sigma, 3 * sigma, kernel_size)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    kernel /= np.sum(kernel)

    pad_width = kernel_size // 2
    F_padded = np.pad(F, pad_width, mode='edge')
    F_smoothed = np.convolve(F_padded, kernel, mode='same')
    F_smoothed = F_smoothed[pad_width:-pad_width]

    return x_grid, F_smoothed


# ************************************
# Function for measuring the relation in the tail of histogram
# ************************************
def exponential_func(x, a, b):
    """ y = a * exp(b*x) """
    return a * np.exp(b * x)


# ************************************
# Main, Parameters, and plots
# ************************************
def main():
    # -------------
    # Parameters
    # -------------
    L = 2
    base_seed = 42  # For a fixed random seed
    J_distribution = "uniform"
    num_realizations = 100
    energy_range_offset = 1.0  # Energy range from E0 to E0+energy_range_offset
    num_samples = 100000  # Monte Carlo sample size
    hist_bins = 100  # Number of bins present in the histogram

    # ************************************
    # Storing and saving the varibales and values
    # ************************************

    realization_counts = []
    first_realization_data = None

    for r in range(num_realizations):
        current_seed = base_seed + r
        print(f"\n=== Realization {r + 1} (Seed: {current_seed}) ===")

        # Generating J and computing Hamiltonians (with a scaling factor)
        J = generate_J_matrix_3D(L, seed=current_seed, J_distribution=J_distribution)
        hamiltonians, _, scale_factor = compute_hamiltonians(J, num_samples=num_samples)

        # Ground state energy
        ground_state_energy = find_ground_state(hamiltonians)
        print(f"Ground State Energy: {ground_state_energy:.4f}")

        # Smoothed cumulative function
        x_grid, F_smoothed = compute_smoothed_cumulative(
            hamiltonians, scale_factor,
            sigma=0.1, num_points=300, extra_margin=0.2
        )

        if r == 0:  # Plotting the desired realization e.g r==0 produces the first realization's DOS and cumulative function
            first_realization_data = (x_grid, F_smoothed)

        # Number of states in [E_ground, E_ground + energy offset]
        E_low = ground_state_energy
        E_high = ground_state_energy + energy_range_offset
        n_low = np.interp(E_low, x_grid, F_smoothed)
        n_high = np.interp(E_high, x_grid, F_smoothed)
        n_val = n_high - n_low
        realization_counts.append(n_val)
        print(f"Number of states in [{E_low:.4f}, {E_high:.4f}]: {n_val:.4f}")

    # Converting the values into numpy array
    n_values = np.array(realization_counts)

    # ************************************
    # Plot 1: Bar Chart
    # ************************************

    average_n = np.mean(n_values)
    plt.figure(figsize=(8, 6))
    x_vals = np.arange(1, num_realizations + 1)
    plt.bar(x_vals, n_values, color="skyblue", edgecolor="k", label="n per Realization")
    plt.axhline(y=average_n, color="red", linestyle="--", label=f"Average = {average_n:.2f}")
    plt.xlabel("Realization Number")
    plt.ylabel("Estimated # of States in [E₀, E₀+1]")
    plt.title("Number of States in Energy Window per Realization")
    plt.xticks(x_vals)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Saving the bar chart (PNG or PDF)
    plt.savefig("bar_chart_n_values.png", dpi=300, bbox_inches="tight")
    # or plt.savefig("bar_chart_n_values.pdf", bbox_inches="tight") could be used if neccesary

    plt.show()

    # ************************************
    # Plot 2: Histogram with Fixed Bins
    # ************************************

    plt.figure(figsize=(8, 6))
    counts, bin_edges, _ = plt.hist(n_values, bins=hist_bins, color="lightgreen", edgecolor="k")
    plt.xlabel("n = Number of States in [E₀, E₀+1]")
    plt.ylabel("Number of Realizations")
    plt.title("Histogram: Realizations vs. Estimated n")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save histogram
    plt.savefig("hist_n_values.png", dpi=300, bbox_inches="tight")
    # or plt.savefig("hist_n_values.pdf", bbox_inches="tight")

    plt.show()

    print(f"\nAverage n over {num_realizations} realizations: {average_n:.4f}")

    # ************************************
    # Tail Fit
    # ************************************
    # Example: Fit the tail of the histogram to an exponential # Power law is tested and find out to be not correct behaviour
    # 1) Convert histogram counts to x=bin_centers, y=counts
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 2) Define a cutoff for the "tail": e.g. top 20% of n_values
    tail_cut = np.percentile(n_values, 80)  # can be adjusted
    mask = bin_centers >= tail_cut

    x_tail = bin_centers[mask]
    y_tail = counts[mask]

    # Avoiding log(0)
    valid_mask = (y_tail > 0)
    x_tail = x_tail[valid_mask]
    y_tail = y_tail[valid_mask]

    if len(x_tail) > 3:
        # Exponential Fit
        try:
            popt_exp, _ = curve_fit(exponential_func, x_tail, y_tail, p0=(1, -0.1))
            print(f"Exponential fit: a={popt_exp[0]:.4f}, b={popt_exp[1]:.4f}")
        except RuntimeError:
            popt_exp = None
            print("Exponential fit failed.")

        # Plotting the tail region + the fits
        plt.figure(figsize=(8, 6))
        plt.scatter(x_tail, y_tail, label="Tail Data", color="blue")

        if popt_exp is not None:
            plt.plot(x_tail, exponential_func(x_tail, *popt_exp),
                     label=f"Exp fit: a={popt_exp[0]:.2f}, b={popt_exp[1]:.2f}",
                     color="red")

        plt.xlabel("Number of states in [E₀, E₀+1] ")
        plt.ylabel("Number of Realizations")
        plt.title("Exponential Law at Tail")
        plt.legend()
        plt.grid(True)

        # Saving the tail-fit figure
        plt.savefig("tail_fit.png", dpi=300, bbox_inches="tight")
        plt.show()
    else:
        print("Not enough tail data points for fitting.")

    # ************************************
    # Smoothed Cumulative & DOS for the Wanted Realization
    # ************************************
    if first_realization_data is not None:
        x_grid, F_smoothed = first_realization_data
        dos = np.gradient(F_smoothed, x_grid)

        plt.figure(figsize=(10, 6))
        plt.plot(x_grid, F_smoothed, label="Smoothed Cumulative F(E)", color="blue")
        plt.xlabel("Energy")
        plt.ylabel("Cumulative # of States")
        plt.title("Smoothed Cumulative Function")
        plt.legend()
        plt.grid(True)
        plt.savefig("desired_realization_cumulative.png", dpi=300, bbox_inches="tight")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(x_grid, dos, label="DOS = dF/dE", color="red")
        plt.xlabel("Energy")
        plt.ylabel("Density of States (DOS)")
        plt.title("Density of States")
        plt.legend()
        plt.grid(True)
        plt.savefig("desired_realization_DOS.png", dpi=300, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    main()
