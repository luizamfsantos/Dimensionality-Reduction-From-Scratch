# Principal Component Analysis
# Using formulas, matrices, and such

# import libraries
import numpy as np

# sample covariance matrix
def calculate_sample_cov_mat(*vectors):
    '''
    Calculate the unbiased sample covariance matrix of a data set

    Parameters
    ----------
    *vectors : numpy.ndarray
        column vectors of the data set
    
    Returns
    -------
    S : numpy.ndarray
        sample covariance matrix
    '''
    # vectors are column vectors
    # Check if they are column vectors
    for vector in vectors:
        if vector.shape[1] != 1:
            print(vector)
            raise ValueError('Vectors must be column vectors')
    # create a matrix of the vectors transpose
    X  = np.hstack(vectors).T
    # check if the matrix is centered if not center it
    if np.allclose(X.mean(axis=0),np.zeros(X.shape[1])):
        X = X - X.mean(axis=0)
    # calculate the number of observations
    n = X.shape[0]
    # calculate the sample covariance matrix
    S = 1/(n-1) * X.T.dot(X)
    return S

# calculate the eigenvalues and eigenvectors of a matrix
def calculate_eigens(A):
    '''
    Calculate the eigenvalues and eigenvectors of a matrix and return them in descending order

    Parameters
    ----------
    A : numpy.ndarray
        matrix
    
    Returns
    -------
    eigenvalues : numpy.ndarray
        eigenvalues of the matrix
    eigenvectors : numpy.ndarray
        eigenvectors of the matrix
    '''
    # calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # sort the eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    # sort the eigenvectors in the same order
    eigenvectors = eigenvectors[:,idx]
    # return the eigenvalues and eigenvectors
    return eigenvalues, eigenvectors
