"""
6D coordinate representation for protein structures (NumPy implementation).

This module provides NumPy-based functions for computing 6D coordinate representations
of protein structures. The 6D representation encodes pairwise geometric relationships
using:
- Distance between C-beta atoms
- Three orientation angles (omega, theta, phi) describing relative orientations

This is an alternative implementation to kinematics.py that uses NumPy instead of PyTorch,
primarily used for preprocessing and analysis tasks outside of the neural network.
"""
import numpy as np
import scipy
import scipy.spatial
from rfdiffusion.kinematics import get_dih

def get_angles(a, b, c):
    """
    Calculate planar angles defined by triplets of 3D points.

    Computes the angle at point b formed by vectors (a-b) and (c-b).

    Args:
        a (np.ndarray): First point set of shape (N, 3)
        b (np.ndarray): Middle point set (vertex) of shape (N, 3)
        c (np.ndarray): Third point set of shape (N, 3)

    Returns:
        np.ndarray: Angles in radians of shape (N,)
    """

    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:,None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:,None]

    x = np.sum(v*w, axis=1)

    #return np.arccos(x)
    return np.arccos(np.clip(x, -1.0, 1.0))

def get_coords6d(xyz, dmax):
    """
    Compute 6D coordinate representation from backbone atom coordinates.

    Converts 3D Cartesian coordinates of backbone atoms (N, CA, C) into a 6D
    representation that captures both distance and orientation information between
    residue pairs. The representation includes:
    - dist6d: C-beta to C-beta distance
    - omega6d: Dihedral angle CA-CB-CB-CA
    - theta6d: Dihedral angle N-CA-CB-CB
    - phi6d: Planar angle CA-CB-CB

    Uses spatial indexing for efficient neighbor searching.

    Args:
        xyz (np.ndarray): Backbone atom coordinates of shape (3, nres, 3) where
            the first dimension corresponds to N, CA, C atoms
        dmax (float): Maximum distance threshold for considering residue pairs

    Returns:
        tuple: (dist6d, omega6d, theta6d, phi6d, mask)
            All arrays are of shape (nres, nres)
            - dist6d: Pairwise C-beta distances
            - omega6d: Dihedral angles CA-CB-CB-CA
            - theta6d: Dihedral angles N-CA-CB-CB
            - phi6d: Planar angles CA-CB-CB
            - mask: Binary mask (1.0 where distance < dmax, 0.0 otherwise)
    """

    nres = xyz.shape[1]

    # three anchor atoms
    N  = xyz[0]
    Ca = xyz[1]
    C  = xyz[2]

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

    # fast neighbors search to collect all
    # Cb-Cb pairs within dmax
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)

    # indices of contacting residues
    idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]

    # Cb-Cb distance matrix
    dist6d = np.full((nres, nres),999.9, dtype=np.float32)
    dist6d[idx0,idx1] = np.linalg.norm(Cb[idx1]-Cb[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((nres, nres), dtype=np.float32)
    omega6d[idx0,idx1] = get_dih(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])
    # matrix of polar coord theta
    theta6d = np.zeros((nres, nres), dtype=np.float32)
    theta6d[idx0,idx1] = get_dih(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])

    # matrix of polar coord phi
    phi6d = np.zeros((nres, nres), dtype=np.float32)
    phi6d[idx0,idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])

    mask = np.zeros((nres, nres), dtype=np.float32)
    mask[idx0, idx1] = 1.0
    return dist6d, omega6d, theta6d, phi6d, mask
