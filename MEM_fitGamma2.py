# MEM_fitGamma2.py
# Reweighting of MD frames using MaxEnt approach to fit mPRE data.
# written by D. Liu

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from scipy.optimize import minimize
from scipy.special import logsumexp


def parse_theta():
    if len(sys.argv) < 2:
        print("Usage: python mem_gama2.py <theta>")
        sys.exit(1)
    return float(sys.argv[1])


def load_experimental_data(file_path):
    """Load experimental mPRE values and errors."""
    data = np.genfromtxt(file_path)
    return data[:, 1:], data[:, 0]  # (values+errors), residue indices


def load_backcalculated_data(file_path, residue_indices):
    """Load simulated mPRE data and extract rows matching experimental residues."""
    raw_data = np.genfromtxt(file_path)
    index_map = {int(row[0]): i for i, row in enumerate(raw_data)}
    selected_indices = [index_map[int(r)] for r in residue_indices]
    return raw_data[selected_indices, 1:].T  # shape: (n_frames, n_observables)


def scale_backcalc_to_experiment(backcalc, weights, experimental_sum):
    """Rescale backcalculated values to match total experimental intensity using current weights."""
    weighted_sum = np.sum(np.dot(backcalc.T, weights))
    return backcalc * (experimental_sum / weighted_sum)


def calc_chi_squared(exp, calc, weights):
    calc_avg = np.sum(calc * weights[:, np.newaxis], axis=0)
    scale = np.sum(exp[:, 0]) / np.sum(calc_avg)
    diff = scale * calc_avg - exp[:, 0]
    return np.average((diff / exp[:, 1]) ** 2)


def maxent_objective(lambdas):
    arg = -np.sum(backcalc * lambdas, axis=1) - MAXN + np.log(w0)
    logz = logsumexp(arg)
    eps2 = 0.5 * np.sum((lambdas * lambdas) * theta_sigma2)
    sum1 = np.dot(lambdas, exp_data[:, 0])
    fun = sum1 + eps2 + logz

    ww = np.exp(arg - logz)
    avg = np.sum(backcalc * ww[:, np.newaxis], axis=0)
    jac = exp_data[:, 0] + lambdas * theta_sigma2 - avg

    return fun / theta, jac / theta


def maxent_hessian(lambdas):
    arg = -np.sum(backcalc * lambdas, axis=1) - MAXN
    ww = w0 * np.exp(arg)
    zz = np.sum(ww)
    if not np.isfinite(zz):
        raise ValueError("Sum of weights is infinite. Consider using a higher theta.")
    ww /= zz
    q_w = np.dot(ww, backcalc)
    hess = np.einsum('k, ki, kj->ij', ww, backcalc, backcalc) - np.outer(q_w, q_w) + np.diag(theta_sigma2)
    return hess / theta


def optimize_weights(calc, exp, theta):
    chi2_history = []
    """
    Optimize frame weights to match experimental mPRE data using maximum entropy.

    Parameters:
        calc (np.ndarray): Back-calculated observables (frames × residues)
        exp (np.ndarray): Experimental data (residues × 2: value, error)
        theta (float): Regularization parameter

    Returns:
        w_opt (np.ndarray): Optimized frame weights
        lambdas (np.ndarray): Lagrange multipliers
    """
    theta_sigma2 = theta * exp[:, 1] ** 2
    w0 = np.ones(calc.shape[0])
    w_uniform = np.ones(calc.shape[0]) / calc.shape[0]
    x2_old = calc_chi_squared(exp, calc, w_uniform)
    x2_new = 100.
    threshold = 5e-2
    exps_sum = np.sum(exp[:, 0])

    iteration = 0
    print("Starting optimization loop...")
    while abs(x2_old - x2_new) > threshold:
        calc = scale_backcalc_to_experiment(calc, w_uniform, exps_sum)
        lambdas = np.zeros(exp.shape[0], dtype=np.longdouble)
        opt = {'maxiter': 50000, 'disp': False}
        result = minimize(
            maxent_objective, lambdas,
            jac=True,
            hess=maxent_hessian,
            method="trust-constr",
            options=opt
        )
        arg = -np.sum(result.x[np.newaxis, :] * calc, axis=1) - MAXN
        w_opt = w0 * np.exp(arg)
        w_opt /= np.sum(w_opt)

        x2_old = calc_chi_squared(exp, calc, w_uniform)
        x2_new = calc_chi_squared(exp, calc, w_opt)
        print(f"Iteration {iteration}: chi2_before = {x2_old:.4f}, chi2_after = {x2_new:.4f}")
        chi2_history.append((iteration, x2_old, x2_new))
        w_uniform = w_opt
        iteration += 1

        save_chi2_history(chi2_history)
    return w_opt, result.x


def save_chi2_history(history, outdir="dist10220finaltest"):
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(history, columns=["Iteration", "Chi2_Before", "Chi2_After"])
    df.to_csv(os.path.join(outdir, "chi2_history.csv"), index=False)

    # Plotting chi² convergence curve
    plt.figure()
    iterations = df["Iteration"]
    chi2_values = df["Chi2_After"]
    plt.plot(iterations, chi2_values, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Chi² After Reweighting")
    plt.title("Convergence of Chi²")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "chi2_convergence.png"), dpi=300)
    plt.close()
    df = pd.DataFrame(history, columns=["Iteration", "Chi2_Before", "Chi2_After"])
    df.to_csv(os.path.join(outdir, "chi2_history.csv"), index=False)

def save_results(weights, lambdas, theta, outdir="output"):
    os.makedirs(outdir, exist_ok=True)
    np.savetxt(f"{outdir}/weights_theta_{theta:.3f}.txt", weights, fmt="%.21f")
    np.savetxt(f"{outdir}/lambdas_theta_{theta:.3f}.txt", lambdas, fmt="%.10f")


def main():
    global MAXN,w0,exp_data,backcalc,theta_sigma2,theta
    theta = parse_theta()
    MAXN = np.log(sys.float_info.max / 5.)
    exp_data, residue_indices = load_experimental_data("mPRE_KRas185GDP.txt.oldmodi")
    theta_sigma2 = theta * exp_data[:,1] ** 2
    backcalc = load_backcalculated_data("../dist10220final/mPRE_complex3t_dist_40tho.data", residue_indices)
    w0 = np.ones(backcalc.shape[0])
    weights, lambdas = optimize_weights(backcalc, exp_data, theta)
    save_results(weights, lambdas, theta, outdir="dist10220finaltest")


if __name__ == "__main__":
    main()
