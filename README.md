# [M.EGI018] Capacity-Pricing Model - Simulated Annealing

**Author(s):**  
XXX | upXXXXXXXX  
João Pedro de Lima Silveira | up202503099  
Matheus Fernandes Vilhena Campinho | 202202004  
XXX | upXXXXXXXX  
XXX | upXXXXXXXX  

**Project Description**

This repository contains the implementation of a **Simulated Annealing metaheuristic** to solve the problem of integrating **capacity** and **pricing** decisions for **car rental companies** (RManagt). The goal is to **maximize** the company's profitability, considering fleet sized and mix, acquisitions, removals, fleet deployment and repositioning, as well as pricing strategies for different rental requests.

The problem addresses the integration of capacity and pricing decisions for car rental companies. This includes:

  - **Fleet Size and Mix:** Decisions on how many vehicles of each type will compose the fleet.
  - **Acquisitions and Removals:** Planning the purchase of owned vehicles and the use of leased vehicles for demand peaks.
  - **Fleet Deployment and Repositioning:** Distribution of vehicles among locations and how they are repositioned to meet demand (including "empty transfers").
  - **Pricing Strategies:** Definition of prices for different rental types, considering the antecedence of the request and its impact on demand.

The underlying mathematical model is an **Integer Non-Linear Programming (INLP)** Formulation, with the objective of maximizing the company's profit over a time horizon. Profit is the difference between revenues from fulfilled rentals and the costs of leasing/acquiring the fleet, empty transfers, maintaining the owned fleet, and a penalty factor for upgrades.

The main **constraints** include:

1.  **Stock Calculating Constraints:** Compute the stock of vehicles of each group in each time period and station.
2.  **Capacity/Demand Constraints:** Establish a limitation on the number of fulfilled rentals and empty transfers.
3.  **Business-Related Constraints:** Establish limitations regarding possible upgrades and available purchase budget.
4.  **Other Auxiliary Constraints.**

**Solution Approach:** Simulated Annealing
The Simulated Annealing (SA) metaheuristic was chosen to solve this combinatorial optimization problem due to its effectiveness in escaping local optima and finding high-quality solutions in complex problems.

**Key Concepts of Simulated Annealing**

  - **Stochastic Algorithm:** Allows for the degradation of a solution to escape local optima.
  - **Memoryless:** Does not use any information gathered during the search.
  - **Iterations:** Proceeds in several iterations, generating a random neighbor at each step.
  - **Move Acceptance:**
      - Moves that improve the cost function are always accepted.
      - Moves that worsen the cost function are accepted with a probability that depends on the current control parameter (`T`) and the degradation (`ΔE`) of the objective function.
  - **Control Parameter (Temperature `T`):** Determines the probability of accepting non-improving solutions. Decreases gradually according to a **cooling schedule**.
  - **Cooling Schedule:** Defines the temperature at each step of the algorithm and has a great impact on SA's performance. It includes:
      - **Initial Temperature:** Should be high enough to allow moves to almost any neighboring state, but not so high as to conduct a random search for too long.
      - **Equilibrium State:** A sufficient number of transitions (moves) must be applied at each temperature to reach an equilibrium state.
      - **Cooling Function:** Temperature `T` is gradually decreased (linear, geometric, logarithmic, etc.).
      - **Final Temperature / Stopping Criterion:** The search can be stopped when the probability of accepting a move is negligible, reaching a low final temperature, or after a predetermined number of iterations without improvement.

Specific Design Elements for SA
In addition to the common concepts of single-solution based metaheuristics (S-metaheuristics), such as neighborhood definition and initial solution generation, the main specific design elements for SA are:

1.  **Acceptance Probability Function:** Enables non-improving neighbors to be selected. The probability is proportional to temperature `T` and inversely proportional to the change in the objective function `ΔE`.
2.  **Cooling Schedule:** Defines the temperature at each step of the algorithm and is crucial for efficiency and effectiveness.

**Implementation**
The implementation will follow the guidelines for metaheuristics, focusing on:

  - **Solution Representation:** How a solution to the capacity-pricing problem is encoded.
  - **Objective Function:** The function that assigns a real value to each solution, representing the quality of the solution (company's profit).
  - **Constraint Handling:** How the mathematical model's constraints are incorporated into solution evaluation or neighbor generation.
  - **Neighborhood Definition:** How the neighbors of a solution are generated.
  - **Initial Solution:** Strategies for generating the initial solution (random or greedy).
  - **Parameter Tuning:** Careful configuration of SA parameters (initial temperature, cooling schedule, stopping criterion) to optimize performance.

**Data**
Instances for testing the heuristic can be downloaded from <https://doi.org/10.17632/g49smv7nh8.1>. This dataset contains 40 instances for the Capacity-Pricing Model for car rental companies (Oliveira et al., 2017).

**References**

  - Oliveira, B. B., M. A. Carravilla, and J. F. Oliveira. 2018. "Integrating Pricing and Capacity Decisions in Car Rental: A Matheuristic Approach." *Operations Research Perspectives* 5: 334–56. <https://doi.org/10.1016/j.orp.2018.10.002>.
  - Oliveira, Beatriz; Carravilla, Maria Antónia; Oliveira, José Fernando (2017), "Capacity-Pricing Model: car rental instances", Mendeley Data, V1, doi: [10.17632/g49smv7nh8.1](https://doi.org/10.17632/g49smv7nh8.1).
