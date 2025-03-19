import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Jmatrix(L, seed=None, Jdistr="uniform"):
    if seed is not None:
        np.random.seed(seed)
    N = L ** 3
    J = np.zeros((N, N))

    def index(x, y, z):
        return ((x % L) * (L ** 2)) + ((y % L) * L) + (z % L)

    boundaries = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = index(x, y, z)
                for dx, dy, dz in boundaries:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    j = index(nx, ny, nz)
                    if i < j:
                        if Jdistr == "uniform":
                            val = np.random.uniform(-1, 1)
                        else:
                            val = np.random.normal(0, 1)
                        J[i, j] = val
                        J[j, i] = val  # Enforcing symmetry
    return J

def compHamiltonians(J, numMC=10000):
    N = J.shape[0]
    totStates = 2 ** N

    def compH(spins):
        return -0.5 * np.sum(J * np.outer(spins, spins))  # 1/2 avoids double-counting

    if totStates <= numMC:
        hamiltonians = np.empty(
            totStates)  # Exact enumaration if the system size is small, otherwise Monte Carlo Sampling
        configurations = []
        for state_int in range(totStates):
            spins = np.array([1 if (state_int >> i) & 1 else -1 for i in range(N)])
            H = compH(spins)
            hamiltonians[state_int] = H
            configurations.append(spins)
        scaleFactor = 1.0
        return hamiltonians, configurations, scaleFactor
    else:
        hamiltonians = np.empty(numMC)
        configurations = []
        for i in range(numMC):
            spins = np.random.choice([1, -1], size=N)
            H = compH(spins)
            hamiltonians[i] = H
            configurations.append(spins)
        scaleFactor = totStates / numMC
        return hamiltonians, configurations, scaleFactor

def groundState(hamiltonians):
    return np.min(hamiltonians)

def cumulativeFunction(hamiltonians, scaleFactor, sigma=0.1, num_points=300, extra_margin=0.2):
    Esort = np.sort(hamiltonians)
    E_min = Esort[0]
    E_max = Esort[-1]
    x_grid = np.linspace(E_min - extra_margin, E_max + extra_margin, num_points)

    Fsamp = np.array([np.searchsorted(Esort, E, side='right') for E in x_grid])
    F = Fsamp * scaleFactor  # scale to represent full space

    kernelSize = 100
    kernel_x = np.linspace(-3 * sigma, 3 * sigma, kernelSize)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    kernel /= np.sum(kernel)

    padWidth = kernelSize // 2
    Fpad = np.pad(F, padWidth, mode='edge')
    Fsmth = np.convolve(Fpad, kernel, mode='same')
    Fsmth = Fsmth[padWidth:-padWidth]
    return x_grid, Fsmth

def exponentialFunction(x, a, b):
    return a * np.exp(b * x) # a is the constant and b is power: ae^(bx)

def main():

    """
    Main part of the code, in future there will be sliders instead of these fixed parameters. Change at will to observe
    the affects of these parameters. For more accurate measurements keep numMC and numRealizations high. Note that at higher
    L's the time for the calculation will grow exponentially, and will require tremendous amounts of system power. This code is
    intended for L's smaller than 7-8 as any bigger than this will cause days of operations to observe the phonemena. Estimated number of
    realiations for accuracy is 10,000+.
    """
    L = 2
    base_seed = 42  # For a fixed random seed
    Jdistr = "uniform"
    numRealizations = 10000
    EnergyOffset = 1.0  # Energy range from E0 to E0+EnergyOffset
    numMC = 100000  # Monte Carlo sample size
    hist_bins = 100  # Number of bins present in the histogram

    realizationCounts = []
    nthRealizationData = None

    for r in range(numRealizations):
        current_seed = base_seed + r

        # Generating J and computing Hamiltonians (with a scaling factor)
        J = Jmatrix(L, seed=current_seed, Jdistr=Jdistr)
        hamiltonians, _, scaleFactor = compHamiltonians(J, numMC=numMC)

        # Ground state energy
        Eground = groundState(hamiltonians)
        # Smoothed cumulative function
        x_grid, Fsmth = cumulativeFunction(
            hamiltonians, scaleFactor,
            sigma=0.1, num_points=300, extra_margin=0.2
        )
        if r == 0:  # Plotting the desired realization e.g r==0 produces the first realization's DOS and cumulative function
            nthRealizationData = (x_grid, Fsmth)

        # Number of states in [E_ground, E_ground + energy offset]
        Elow = Eground
        Ehigh = Eground + EnergyOffset
        n_low = np.interp(Elow, x_grid, Fsmth)
        n_high = np.interp(Ehigh, x_grid, Fsmth)
        n_val = n_high - n_low
        print(f"Realization {r + 1}: E_ground = {Eground:.4f}, States in range: {n_val:.4f}")
        realizationCounts.append(n_val)
    nVal = np.array(realizationCounts)

    nAvg = np.mean(nVal)
    plt.figure(figsize=(8, 6))
    x_vals = np.arange(1, numRealizations + 1)
    plt.bar(x_vals, nVal, color="skyblue", edgecolor="k", label="n per Realization")
    plt.axhline(y=nAvg, color="red", linestyle="--", label=f"Average = {nAvg:.2f}")
    plt.xlabel("Realization Number")
    plt.ylabel("Estimated # of States in [E₀, E₀+1]")
    plt.title("Number of States in Energy Window per Realization")
    plt.xticks(x_vals)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Saving the bar chart as PNG or PDF
    plt.savefig("bar_chart_nVal.png", dpi=300, bbox_inches="tight")
    # or plt.savefig("bar_chart_nVal.pdf", bbox_inches="tight") could be used if neccesary
    plt.show()

    plt.figure(figsize=(8, 6))
    counts, bin_edges, _ = plt.hist(nVal, bins=hist_bins, color="lightgreen", edgecolor="k")
    plt.xlabel("n = Number of States in [E₀, E₀+1]")
    plt.ylabel("Number of Realizations")
    plt.title("Histogram: Realizations vs. Estimated n")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save histogram
    plt.savefig("hist_nVal.png", dpi=300, bbox_inches="tight")
    # or plt.savefig("hist_nVal.pdf", bbox_inches="tight")

    plt.show()

    print(f"\nAverage n over {numRealizations} realizations: {nAvg:.4f}")

    bC = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    tailCut = np.percentile(nVal, 80)  # can be adjusted
    mask = bC >= tailCut

    xTail = bC[mask]
    yTail = counts[mask]

    # For Avoiding log(0)
    valid_mask = (yTail > 0)
    xTail = xTail[valid_mask]
    yTail = yTail[valid_mask]

    if len(xTail) > 3:
        # Exponential Fit
        popt_exp, _ = curve_fit(exponentialFunction, xTail, yTail, p0=(1, -0.1))
        print(f"Exp fit: a={popt_exp[0]:.4f}, b={popt_exp[1]:.4f}")

        # Plotting the tail region + the other fits
        plt.figure(figsize=(8, 6))
        plt.scatter(xTail, yTail, label="Tail Data", color="blue")

        if popt_exp is not None:
            plt.plot(xTail, exponentialFunction(xTail, *popt_exp),
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

    if nthRealizationData is not None:
        x_grid, Fsmth = nthRealizationData
        dos = np.gradient(Fsmth, x_grid)

        plt.figure(figsize=(10, 6))
        plt.plot(x_grid, Fsmth, label="Smoothed Cumulative F(E)", color="blue")
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
