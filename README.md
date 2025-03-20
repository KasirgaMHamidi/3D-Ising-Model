# 3D-Ising-Model
Simulating the Behaviors of Ising Glasses in 3-Dimensions

Ising Model for 1 and 2 dimensions are well studied and already investigated until their most secret corner. There is not much left to learn from it at this point of time. Therefore, we need to study a higher, more realistic dimension, 3-Dimensional Ising Model. Studying this model in three dimensions will undoubtedly shed light on some of the intricate problems regarding phase transitions and ferromagnetism. 

However, in this study I'm not goint to investigate the ferromagnetism just yet. The main purpose of this work is to observe how the number of states near the ground state energy changes as the system size increases, universal scalings of the distribution of states, and quantum annealing, or in other words, optimization of the larger picture. Due to complex background of the system, mainly arising from the coupling matrix $J_{ij}$, phase transitions are not in the target board. 

The System is a 3D Cubic lattice, where at each lattice site i there is an atom or molecule that take the values either -1 or 1, -1 indicating down spin and 1 indicating up spin. On another nore, the system's boundary lattice points also interact with the one at the opposite site, effectively making a thorus shaped interaction field. To diminish the affects of the double counting the Hamiltonian, the total energy of the system, is divided by two, and it is given as

$$
H = -\frac{1}{2} \sum_{i,j} J_{ij} \sigma_i \sigma_j
$$

where $J_{ij}$ represents the interaction strength between spins $\sigma_i$ and $\sigma_j$. Afterwards, for maximum effectivity and optimization, Monte Carlo or Metropolis algorithm is used. User can input different values of Monte Carlo Sweeps to enhance or reduce the precision and accuracy of the simulation. It must be also noted that the simulation is only effective until L size (L represent the number of lattice points in one side of the thorus) of 6 or 7, after this threshold accuracy drops significantly due to optimization problems in the code. 

Lastly, the simulation measures the decay function in the tail of the number of realizations having that number of states vs. number of states at the given energy range near the ground state as an exponential relation.

Here are the results of the 2x2x2 3D Ising Model with Periodic Boundary Condtions, 10k+ realizations, and 100k monte carlo sweeps (the simulation uses exact enumaration if the number of Monte Carlo sweeps is equal or larger than $2^{L^3}$:

![Smoothed Cumulative Function](https://github.com/user-attachments/assets/32541c03-0a82-491c-8696-b62ca4858223)
![DOS_Smooth](https://github.com/user-attachments/assets/e1ca349a-4dbd-466a-8cef-4532ca2ea642)
![hist_n_values](https://github.com/user-attachments/assets/315d19b5-bee9-4123-b120-c6e85a2c36d6)
![tail_fit](https://github.com/user-attachments/assets/f0e3341b-87f4-46f2-b43b-03a3239c6842)

Note: The simulation calculates all of the plots above by first constructing a step cumulative function of the Hamiltonian, than smoothing it out by gaussians and kernels to remove so to become delta functions (discontinuties) at the Density of States (DOS) graph. First graph is the smoothed out step function which than made into a DOS, than with different $J_{ij}$'s over many realizations made into a histogram, and lastly the exponential function found for the rare-to-found energy states of the LxLXL system.
