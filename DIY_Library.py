import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models, optimizers, losses
from tqdm import tqdm
from math import isclose


def Hamiltonian(qx, qy, px, py):
    return 0.5*(px**2 + py**2) + 0.5*(qx**2 + qy**2) + (qx**2 * qy - (1.0/3.0)*qy**3)

def Hamiltonian_ODE_solve(t, z):
    qx, qy, px, py = z
    dq_x = px
    dq_y = py
    dp_x = -qx - 2*qx*qy
    dp_y = -qy - qx**2 + qy**2
    return [dq_x, dq_y, dp_x, dp_y]

def sample_initial_condition(E_target, rng=np.random):
    while True:
        qx = rng.uniform(-0.6, 0.6)
        qy = rng.uniform(-0.6, 0.6)
        V = 0.5*(qx*qx + qy*qy) + (qx*qx * qy - (1.0/3.0)*qy**3)
        if V < E_target:
            break
    K = max(E_target - V, 0.0)
    theta = rng.uniform(0, 2*np.pi)
    p_mag = np.sqrt(2*K)
    px = p_mag * np.cos(theta)
    py = p_mag * np.sin(theta)
    return np.array([qx, qy, px, py])

def generate_exact_dataset(Emin=0.001, Emax=1/6 - 1e-6, n_energies=15, orbits_per_energy=20, orbit_time=5000.0, dt=0.1):
    rng = np.random.RandomState(42)
    energies_list = np.linspace(Emin, Emax, n_energies)

    X_all = []
    dX_all = []
    energy_all = []

    for E in tqdm(energies_list, desc="Energy levels"):
        for _ in tqdm(range(orbits_per_energy), leave=False):
            
            # Sample IC at this energy
            z0 = sample_initial_condition(E)

            # Integration setup
            t_span = (0.0, orbit_time)
            t_eval = np.arange(0, orbit_time + dt, dt)  # 50,000 steps

            # Solve using RK45
            sol = solve_ivp(
                Hamiltonian_ODE_solve, 
                t_span, 
                z0, 
                t_eval=t_eval,
                rtol=1e-9, 
                atol=1e-12
            )

            Z = sol.y.T  # (steps+1, 4)

            # Time derivatives
            dZ = np.array([Hamiltonian_ODE_solve(None, z) for z in Z])

            # Energies
            Ener = np.array([Hamiltonian(*z) for z in Z])

            X_all.append(Z[:-1])
            dX_all.append(dZ[:-1])
            energy_all.append(Ener[:-1])

    X = np.concatenate(X_all)
    dX = np.concatenate(dX_all)
    energy = np.concatenate(energy_all)

    return X.astype(np.float32), dX.astype(np.float32), energy.astype(np.float32)

def ANN(input_dim=4, neurons=[200,200]):
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for u in neurons:
        x = layers.Dense(u, activation = "tanh")(x)
    A_out = layers.Dense(4, activation=None)(x)
    model = models.Model(inputs=inputs, outputs=A_out, name ="ANN")
    return model

def HNN(input_dim=4, neurons=[200,200]):
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for u in neurons:
        x = layers.Dense(u, activation = "tanh")(x)
    H_out = layers.Dense(1, activation=None)(x)
    model = models.Model(inputs=inputs, outputs=H_out, name ="HNN")
    return model






