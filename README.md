# Newton-Raphson Power Flow Calculation Algorithm
This repository contains a Python-based implementation of the **Newton-Raphson (N-R) method**, a powerful iterative algorithm used to solve the power flow problem in electrical engineering. 
This project focuses on translating complex grid equations into a functional computational tool for power system analysis.

## Project Overview
The power flow (load flow) problem is essential for determining the steady-state operation of an electric power system. Since the relationship between voltage and power is non-linear, this project utilizes the Newton-Raphson method for its robust and fast convergence characteristics.

### Key Features
* **Automated $Y_{bus}$ Construction**: Builds the Admittance Matrix from line and bus data.
* **Full Jacobian Implementation**: Dynamically updates the Jacobian matrix ($J_1$ through $J_4$) in each iteration.
* **Flexible Bus Handling**: Supports Slack, PV (Generator), and PQ (Load) bus types.
* **Mismatch Analysis**: Calculates real ($P$) and reactive ($Q$) power mismatches to ensure system balance.

---

## Mathematical Framework
The algorithm iteratively solves for unknown state variables (Voltage Magnitude $|V|$ and Phase Angle $\delta$) using the following linearized relationship :

$$\begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix} = \begin{bmatrix} J_1 & J_2 \\ J_3 & J_4 \end{bmatrix} \begin{bmatrix} \Delta \delta \\ \Delta |V| \end{bmatrix}$$

Convergence is achieved when the maximum mismatch ($\Delta P_{max}, \Delta Q_{max}$) falls below a predefined tolerance (e.g., $10^{-6}$).

---

## Data & Implementation

The project processes data through two main structures:
1. **Line Data (`line_df`)**: Contains branch parameters ($R, X, B/2$).
2. **Bus Data (`bus_df`)**: Contains nodal information (Bus type, $P_{gen}, Q_{gen}, P_{load}, Q_{load}$).

### Execution Workflow:
1. **Data Loading**: Import system data via `data_loader.py`.
2. **Matrix Initialization**: Formulate the $Y_{bus}$ matrix.
3. **Iteration Loop**: Compute power mismatches $\rightarrow$ Update Jacobian $\rightarrow$ Solve for corrections $\rightarrow$ Update $V$ and $\delta$.
4. **Final Calculation**: Determine line flows and total system losses.

---

## Results & Validation
Upon successful execution, the algorithm provides a detailed summary of the system state.

### 1. Convergence Performance
The Newton-Raphson method typically converges within 3-5 iterations for standard IEEE test systems. You can verify the convergence by checking the mismatch values in the terminal output.

### 2. Output Summary
The solver outputs a final report including:
* **Bus Voltages**: Magnitude (p.u.) and Angle (degrees).
* **Power Flow**: Real and Reactive power flowing through each line.
* **System Losses**: Total $I^2R$ losses across the network.

> **Note** : You can visualize the convergence by plotting the maximum mismatch vs. the number of iterations to observe the quadratic convergence characteristic.

---

## Getting Started

### Prerequisites
* Python 3.x
* NumPy
* Pandas

### Installation & Run
```bash
# Clone the repository
git clone [https://github.com/sumiiP/PSL_power_flow.git](https://github.com/sumiiP/PSL_power_flow.git)

# Install dependencies
pip install numpy pandas

# Run the power flow solver
python main.py
