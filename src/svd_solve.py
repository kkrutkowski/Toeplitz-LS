import numpy as np

def svd_one_sided_jacobi(A, tol=1e-10, max_sweeps=20):
    m, n = A.shape
    V = np.eye(n)
    B = A.copy()

    for _ in range(max_sweeps):
        converged = True
        for i in range(n):
            for j in range(i+1, n):
                # Compute inner products of columns i and j
                p = B[:, i].T @ B[:, j]
                q_i = B[:, i].T @ B[:, i]
                q_j = B[:, j].T @ B[:, j]

                if abs(p) < tol * np.sqrt(q_i * q_j):
                    continue          # already orthogonal enough
                converged = False

                # Compute Jacobi rotation to orthogonalise columns i, j
                tau = (q_j - q_i) / (2 * p)
                t = np.sign(tau) / (abs(tau) + np.sqrt(1 + tau**2))
                c = 1 / np.sqrt(1 + t**2)
                s = t * c

                # Apply rotation to columns of B
                Bi = B[:, i].copy()
                Bj = B[:, j].copy()
                B[:, i] = c * Bi + s * Bj
                B[:, j] = -s * Bi + c * Bj

                # Update V
                Vi = V[:, i].copy()
                Vj = V[:, j].copy()
                V[:, i] = c * Vi + s * Vj
                V[:, j] = -s * Vi + c * Vj
        if converged:
            break

    # Form Sigma and U
    sigma = np.linalg.norm(B, axis=0)   # column norms -> singular values
    # Avoid division by zero for zero singular values
    U = np.zeros((m, n))
    for j in range(n):
        if sigma[j] > tol:
            U[:, j] = B[:, j] / sigma[j]
    # Extra columns of U if m > n (arbitrary orthogonal extension – omitted for minimality)
    return U, sigma, V.T

def svd_solve(A, b, tol=1e-12):
    U, s, Vt = svd_one_sided_jacobi(A)
    # Pseudo-inverse of Sigma
    s_inv = np.array([1/si if si > tol else 0.0 for si in s])
    # x = V * Sigma^+ * U^T * b
    return Vt.T @ (s_inv * (U.T @ b))
