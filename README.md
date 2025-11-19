# Physics-Enhanced Neural Networks for Chaotic Hamiltonian Dynamics

This is a project that explores how ordinary neural networks (ANNs) and Hamiltonian Neural Networks (HNNs) learn and predict the dynamics of the chaotic Hénon–Heiles Hamiltonian system. It compares their behavior by training on simulated trajectories and predicting for random initial conditions for various energies.

---
## Overview
- Generate trajectories using numerical ODE solvers (used Scipy.solve_ivp(), 'based on RK45')  
- Train:
  - An **Artificial neural network (ANN)** which predicts \(\dot q, \dot p\)  
  - A **Hamiltonian Neural Network (HNN)** that predicts the scalar Hamiltonian and computes derivatives to find \(\dot q, \dot p\) via gradients
- Perform **Trajectory and Energy Drift comparisons** between ANN, HNN, and RK4 (as per code written in library in P-345 Computational Physics Course)  
- Plotting and analysis:
  - Phase-space trajectories (q_y vs q_x) 
  - Energy drift / conservation 
  - Stability of HNN and Divergence of ANN  
