import numpy as np
import matplotlib.pyplot as plt
import time


def generate_synthetic_data_case1(n, m, seed=None):
    np.random.seed(seed)

    A = np.random.randn(n - 1, m)  # Standard normal entries for the first n-1 rows
    A_last_row = np.random.normal(0, np.sqrt(10 ** 2),
                                  m)  # Normal entries with mean 0 and variance 10^2 for the last row
    A = np.vstack((A, A_last_row))

    x0 = np.random.randn(m)
    noise = np.random.normal(0, np.sqrt(0.1 ** 2), n)  # Normal entries with mean 0 and variance 0.1^2
    b = np.dot(A, x0) + noise

    return A, b


def generate_synthetic_data_case2(n, m, seed=None):
    np.random.seed(seed)

    A = np.random.randn(n, m)  # Standard normal entries for all rows

    x0 = np.random.randn(m)
    noise = np.random.normal(0, np.sqrt(0.1 ** 2), n)  # Normal entries with mean 0 and variance 0.1^2
    b = np.dot(A, x0) + noise

    return A, b


def calculate_step_size(mu, epsilon, lambd, L, sigma):
    term1 = mu * epsilon
    term2 = 2 * epsilon * mu * min(np.mean(L) / (1 - lambd), max(L) / lambd)
    term3 = 2 * max(1 / lambd, np.mean(L) / ((1 - lambd) * min(L)))
    gamma = term1 / (term2 + term3)
    return gamma


def calculate_sigma(A, b, x_star):
    norms_squared = []
    for i in range(A.shape[0]):
        gradient = 2 * (np.dot(A[i], x_star) - b[i]) * A[i]
        norms_squared.append(np.linalg.norm(gradient))

    # Estimate sigma squared as the average of the squared norms
    sigma_squared = np.mean(norms_squared)

    return sigma_squared


def sgd_with_partially_biased_sampling(A, b, weights, mu, epsilon, max_iterations=1000, lambd=0):
    n, m = A.shape
    x = np.zeros(m)
    history = []

    x_star = np.linalg.lstsq(A, b, rcond=None)[0]
    L = n * np.linalg.norm(A, axis=1) ** 2
    sigma = calculate_sigma(A, b, x_star)
    step_size = calculate_step_size(mu, epsilon, lambd, L, sigma)

    for iteration in range(1, max_iterations + 1):
        i = np.random.choice(n, p=weights / np.sum(weights))
        gradient = 2 * (np.dot(A[i], x) - b[i]) * A[i]
        x = x - step_size * gradient / weights[i]

        # Record the distance to the true solution for each iteration
        distance = np.linalg.norm(x - x_star)
        history.append(distance)

    return history


def plot_sgd_performance(ax, A, b, lambda_vals, variance, title):
    mu = 1 / np.linalg.norm(np.linalg.inv(np.dot(A.T, A)))  # Strong convexity parameter
    epsilon = 1e-1
    max_iterations = 100000

    for lambda_val in lambda_vals:
        weights = lambda_val + (1 - lambda_val) * np.linalg.norm(A, axis=1) / np.mean(np.linalg.norm(A, axis=1))
        distances = sgd_with_partially_biased_sampling(A, b, weights, mu, epsilon, max_iterations, lambda_val)
        ax.plot(distances, label=f'λ = {lambda_val}')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Distance to True Solution')
    ax.set_title(title)
    ax.legend()


seed = 2
# Set up the synthetic problems
n = 1000  # Adjust as needed
m = 10
lambda_values = [0.99, 0.7, 0.5, 0.2, 0.01]

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Scenario 1
A1, b1 = generate_synthetic_data_case1(n, m, seed=seed)
plot_sgd_performance(axs[0, 0], A1, b1, lambda_values, 0.1, 'Scenario 1')

# Scenario 2
A2, b2 = generate_synthetic_data_case2(n, m, seed=seed)
plot_sgd_performance(axs[0, 1], A2, b2, lambda_values, 0.1, 'Scenario 2')

plt.tight_layout()
plt.show()
