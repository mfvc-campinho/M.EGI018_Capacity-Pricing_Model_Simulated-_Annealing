# [M.EGI018] Capacity-Pricing Model - Simulated Annealing

**Author(s):**  

Daniel Pereira | up202107356  
João Pedro de Lima Silveira | up202503099  
XXX | upXXXXXXXX  
Matheus Fernandes Vilhena Campinho | 202202004  
Rui Rodrigues | up202105516

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
