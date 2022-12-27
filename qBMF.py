import numpy as np
import dimod
import random
from scipy.stats import bernoulli


def construct_BMF_BQM(A, r, lam=None, format="bqm", formulation=1):
    """Construct QUBO formulation for binary matrix factorization (BMF).
    
    This code is based on the paper [MUR+21]. It formulates the problem of 
    computing the rank-r BMF of A as a QUBO. The variable `lam` controls the
    constraint penalty. If it is large enough, the minimum of the QUBO 
    corresponds to the optimal BMF solution. The function outputs either a 
    dictionary containing the non-zero entries in the QUBO matrix, or a 
    dimod.BQM object containing the QUBO. The function can output the first or 
    second QUBO formulation in [MUR+21].

    Args: 
        A: 
            A matrix (np.array) with binary entries.
        r:
            The target decomposition rank.
        lam:
            The penalty used for the constraints; see [MUR+21] for details.
        format:
            Either "bqm" (default) or "dict".
        formulation:
            Either 1 (default) or 2.

    Returns:
        Returns either a dictionary whose keys are the nonzero entries in the 
        QUBO matrix and whose values are the values of those nonzero entries, or
        a dimod.BQM object which encodes the QUBO matrix. This is controlled by 
        the `format` parameter.

    References:
    [MUR+21]    O. A. Malik, H. Ushijima-Mwesigwa, A. Roy, A. Mandal, I. Ghosh. 
                Binary matrix factorization on special purpose hardware. PLOS 
                ONE 16(12): e0261250, 2021. DOI: 10.1371/journal.pone.0261250
    """

    m, n = A.shape
    
    # If no value for lam provided, set it to its default value
    if lam is None:
        lam = 2.1*r*np.linalg.norm(A)

    # Predefine some useful matrices
    I_m = np.identity(m)
    I_r = np.identity(r)
    I_nr = np.identity(n*r)
    one_r = np.ones((r, r))
    one_m_n = np.ones((m, n))
    
    # Use one of two different formulations
    if formulation == 1:
        # Predefine some useful matrices used in formulation 1
        I_mn = np.identity(m*n)
        I_mnr = np.identity(m*n*r)
        one_n_m = np.ones((n, m))
        one_1_m = np.ones((1, m))
        one_1_n = np.ones((1, n))
        one_r_1 = np.ones((r, 1))
        zero_mr = np.zeros((m*r, m*r))
        zero_nr = np.zeros((n*r, n*r))
        vecA = np.reshape(A, (m*n, 1), order='F')

        # Construct Q1
        Q1_upper = np.concatenate((zero_mr, np.kron(I_r, one_m_n)), axis=1)
        Q1_lower = np.concatenate((np.kron(I_r, one_n_m), zero_nr), axis=1)
        Q1 = lam/2 * np.concatenate((Q1_upper, Q1_lower), axis=0)

        # Construct Q2
        Q2_upper = np.kron(np.kron(I_r, one_1_n), I_m)
        Q2_lower = np.kron(I_nr, one_1_m)
        Q2 = -2*lam * np.concatenate((Q2_upper, Q2_lower), axis=0)

        # Construct Q3
        Q3_first = np.kron(one_r, I_mn)
        Q3_second = -2 * np.diag(np.squeeze(np.kron(one_r_1, I_mn) @ vecA))
        Q3_third = 3*lam * I_mnr
        Q3 = Q3_first + Q3_second + Q3_third

        # Construct Q
        Q_upper = np.concatenate((Q1, Q2), axis=1)
        Q_lower = np.concatenate((np.zeros((m*n*r, (m+n)*r)), Q3), axis=1)
        Q = np.concatenate((Q_upper, Q_lower), axis=0)
    
    elif formulation == 2:
        # Predefine some useful matrices used in formulation 2
        I_n = np.identity(n)
        I_mr = np.identity(m*r)
        I_mrr = np.identity(m*r**2)
        I_nrr = np.identity(n*r**2)
        I_rr = np.identity(r**2)
        one_1_r = np.ones((1, r))

        # Construct Q1
        Q1_upper = np.concatenate((np.kron(lam*one_r, I_m), np.kron(-2*I_r, A)), axis=1)
        Q1_lower = np.concatenate((np.zeros((n*r, m*r)), np.kron(lam*one_r, I_n)), axis=1)
        Q1 = np.concatenate((Q1_upper, Q1_lower), axis=0)

        # Construct Q2
        Q2_top_left = np.kron(one_1_r, I_mr) + np.kron(np.kron(I_r, one_1_r), I_m)
        Q2_top_right = np.zeros((m*r, n*r**2))
        Q2_bottom_left = np.zeros((n*r, m*r**2))
        Q2_bottom_right = np.kron(one_1_r, I_nr) + np.kron(np.kron(I_r, one_1_r), I_n)
        Q2 = -2*lam * np.concatenate(
            (np.concatenate((Q2_top_left, Q2_top_right), axis=1),
            np.concatenate((Q2_bottom_left, Q2_bottom_right), axis=1)), 
            axis=0)

        # Construct Q3
        Q3_upper = np.concatenate((3*lam*I_mrr, np.kron(I_rr, one_m_n)), axis=1)
        Q3_lower = np.concatenate((np.zeros((n*r**2, m*r**2)), 3*lam*I_nrr), axis=1)
        Q3 = np.concatenate((Q3_upper, Q3_lower), axis=0)

        # Construct Q
        Q_upper = np.concatenate((Q1, Q2), axis=1)
        Q_lower = np.concatenate((np.zeros(((m+n)*r**2, (m+n)*r)), Q3), axis=1)
        Q = np.concatenate((Q_upper, Q_lower), axis=0)

    # Ensure Q is upper triangular
    Q = np.triu(Q) + np.transpose(np.tril(Q, -1))

    # Turn Q into a dictionary of nonzeros
    Q_dict = {}
    for row in range(Q.shape[0]):
        for col in range(Q.shape[1]):
            if Q[row, col] != 0:
                Q_dict[(row, col)] = Q[row, col]
    
    if format == "dict":  # Return Q as dictionary
        return Q_dict
    elif format == "bqm":  # Return Q as dimod.BQM
        bqm = dimod.BQM.from_qubo(Q_dict)
        return bqm


def extract_U_V(sampleset, m, n, r):
    """Extract the solution matrices U and V from a sampleset.

    Takes a sampleset returned from a D-Wave sampler and returns the U and V 
    matrices corresponding to the lowest energy (i.e., best) solution.

    Args:
        sampleset: 
            A sampleset returned from one of D-Wave's samplers.
        m: 
            The number of rows in the matrix being decomposed.
        n: 
            The number of columns in the matrix being decomposed.
        r:
            The target rank of the decomposition.
    
    Returns:
        Returns the matrices U and V extracted from the lowest energy solution 
        in the sampleset.
    """

    best_samp = sampleset.first.sample
    sol_vec = np.array([best_samp[k] for k in range(len(best_samp))], dtype=int)
    U = np.reshape(sol_vec[:m*r], (m, r), order='F')
    V = np.reshape(sol_vec[m*r:(m+n)*r], (n, r), order='F')
    return U, V


def generate_binary_matrix(m, n, r, pU=0.5, pV=0.5):
    """Generate factors for matrix with exact BMF rank.

    Generates an m-by-n binary matrix which has exact binary rank r. The 
    function returns the factors U and V for the matrix. Minor note: This 
    function is somewhat different from Algorithm 3 in the supplement of 
    [MUR+21].

    Args:
        m:
            Number of rows of binary matrix being created.
        n:
            Number of columns of the binary matrix being created.
        r:
            Target rank r.
        pU:
            Initial density of U matrix. Should be between 0 and 1.
        pV:
            Initial density of V matrix. Should be between 0 and 1.
    
    Returns:
        Returns the matrices U and V of size m-by-r and n-by-r, respectively, so
        that they form the matrix A = U @ np.transpose(V) which is binary and
        has binary rank r.

    References:
    [MUR+21]    O. A. Malik, H. Ushijima-Mwesigwa, A. Roy, A. Mandal, I. Ghosh. 
                Binary matrix factorization on special purpose hardware. PLOS 
                ONE 16(12): e0261250, 2021. DOI: 10.1371/journal.pone.0261250
    """

    # Initialize U and V randomly with predetermined average density
    U = bernoulli.rvs(pU, size=(m, r))
    V = bernoulli.rvs(pV, size=(n, r))
    A = U @ np.transpose(V)

    # Keep running the following until A is binary
    while (A>1).any():
        # Draw random entry in A which exceeds 1
        vecA = A.reshape(m*n)
        ridx = random.choice([k for k in range(m*n) if vecA[k]>1])
        row = ridx // n
        col = ridx % n

        # Determine entry in U or V to set to zero
        r_joint_idx = random.choice([k for k in range(r) if U[row, k]*V[col, k] > 0])
        
        # Set entry in either U or V to zero, depending on which is densest.
        if np.sum(U)/(m*r) > np.sum(V)/(n*r):  
            # U more dense than V, so eliminate nonzero in U
            U[row, r_joint_idx] = 0
        else:
            # V more dense than U, so eliminate nonzero in V
            V[col, r_joint_idx] = 0

        A = U @ np.transpose(V)

    return U, V
