"""
Protein kinematics and structure building utilities.

This module provides functions for computing geometric properties of protein structures
and converting between different representations:
- Distance calculations between atoms and residues
- Angle and dihedral angle computations (phi, psi, omega, chi)
- 6D coordinate representations (distance + 3 orientation angles)
- Binning and discretization of continuous geometric values

These functions are used to prepare structural features for the neural network
and to convert between Cartesian coordinates and internal coordinate representations.
"""
import numpy as np
import torch
from rfdiffusion.chemical import INIT_CRDS
from rfdiffusion.util import generate_Cbeta

# Default parameters for distance and angle binning
PARAMS = {
    "DMIN"    : 2.0,   # Minimum distance in Angstroms
    "DMAX"    : 20.0,  # Maximum distance in Angstroms
    "DBINS"   : 36,    # Number of distance bins
    "ABINS"   : 36,    # Number of angle bins
}

# ============================================================
def get_pair_dist(a, b):
    """calculate pair distances between two sets of points
    
    Parameters
    ----------
    a,b : pytorch tensors of shape [batch,nres,3]
          store Cartesian coordinates of two sets of atoms
    Returns
    -------
    dist : pytorch tensor of shape [batch,nres,nres]
           stores paitwise distances between atoms in a and b
    """

    dist = torch.cdist(a, b, p=2)
    return dist

# ============================================================
def get_ang(a, b, c):
    """calculate planar angles for all consecutive triples (a[i],b[i],c[i])
    from Cartesian coordinates of three sets of atoms a,b,c 

    Parameters
    ----------
    a,b,c : pytorch tensors of shape [batch,nres,3]
            store Cartesian coordinates of three sets of atoms
    Returns
    -------
    ang : pytorch tensor of shape [batch,nres]
          stores resulting planar angles
    """
    v = a - b
    w = c - b
    v /= torch.norm(v, dim=-1, keepdim=True)
    w /= torch.norm(w, dim=-1, keepdim=True)
    vw = torch.sum(v*w, dim=-1)

    return torch.acos(vw)

# ============================================================
def get_dih(a, b, c, d):
    """calculate dihedral angles for all consecutive quadruples (a[i],b[i],c[i],d[i])
    given Cartesian coordinates of four sets of atoms a,b,c,d

    Parameters
    ----------
    a,b,c,d : pytorch tensors or numpy array of shape [batch,nres,3]
              store Cartesian coordinates of four sets of atoms
    Returns
    -------
    dih : pytorch tensor or numpy array of shape [batch,nres]
          stores resulting dihedrals
    """
    convert_to_torch = lambda *arrays: [torch.from_numpy(arr) for arr in arrays]
    output_np=False
    if isinstance(a, np.ndarray):
        output_np=True
        a,b,c,d = convert_to_torch(a,b,c,d)
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1 /= torch.norm(b1, dim=-1, keepdim=True)

    v = b0 - torch.sum(b0*b1, dim=-1, keepdim=True)*b1
    w = b2 - torch.sum(b2*b1, dim=-1, keepdim=True)*b1

    x = torch.sum(v*w, dim=-1)
    y = torch.sum(torch.cross(b1,v,dim=-1)*w, dim=-1)
    output = torch.atan2(y, x)
    if output_np:
        return output.numpy()
    return output

# ============================================================
def xyz_to_c6d(xyz, params=PARAMS):
    """
    Convert Cartesian coordinates to 6D coordinate representation.

    The 6D representation encodes both distance and orientation between residue pairs:
    - dist: C-beta to C-beta distance
    - omega: Dihedral angle CA-CB-CB-CA (orientation around inter-residue axis)
    - theta: Dihedral angle N-CA-CB-CB (local orientation)
    - phi: Planar angle CA-CB-CB (opening angle)

    This representation captures the relative geometry between residues in a way that
    is invariant to global rotations and translations.

    Args:
        xyz (torch.Tensor): Backbone coordinates of shape (batch, nres, 3, 3)
            where the 3 atoms are N, CA, C
        params (dict, optional): Parameters including DMAX for distance cutoff

    Returns:
        tuple: (c6d, mask)
            - c6d: 6D coordinates of shape (batch, nres, nres, 4) containing
                [distance, omega, theta, phi]
            - mask: Binary mask of shape (batch, nres, nres) indicating valid entries
                (1.0 where distance < DMAX, 0.0 otherwise)
    """
    
    batch = xyz.shape[0]
    nres = xyz.shape[1]

    # three anchor atoms
    N  = xyz[:,:,0]
    Ca = xyz[:,:,1]
    C  = xyz[:,:,2]
    Cb = generate_Cbeta(N, Ca, C)

    # 6d coordinates order: (dist,omega,theta,phi)
    c6d = torch.zeros([batch,nres,nres,4],dtype=xyz.dtype,device=xyz.device)

    dist = get_pair_dist(Cb,Cb)
    dist[torch.isnan(dist)] = 999.9
    c6d[...,0] = dist + 999.9*torch.eye(nres,device=xyz.device)[None,...]
    b,i,j = torch.where(c6d[...,0]<params['DMAX'])

    c6d[b,i,j,torch.full_like(b,1)] = get_dih(Ca[b,i], Cb[b,i], Cb[b,j], Ca[b,j])
    c6d[b,i,j,torch.full_like(b,2)] = get_dih(N[b,i], Ca[b,i], Cb[b,i], Cb[b,j])
    c6d[b,i,j,torch.full_like(b,3)] = get_ang(Ca[b,i], Cb[b,i], Cb[b,j])

    # fix long-range distances
    c6d[...,0][c6d[...,0]>=params['DMAX']] = 999.9
    
    mask = torch.zeros((batch, nres,nres), dtype=xyz.dtype, device=xyz.device)
    mask[b,i,j] = 1.0
    return c6d, mask
    
def xyz_to_t2d(xyz_t, params=PARAMS):
    """
    Convert template coordinates to 2D distance and orientation features.

    Processes template structures to create 2D pairwise features suitable for
    input to the neural network. The features include:
    - One-hot encoded distance bins (37 bins)
    - Sinusoidal encodings of orientation angles omega, theta, phi (6 values)
    - Validity mask (1 value)

    Args:
        xyz_t (torch.Tensor): Template backbone coordinates of shape
            (batch, n_templates, nres, 3, 3) where the 3 atoms are N, CA, C
        params (dict, optional): Binning parameters

    Returns:
        torch.Tensor: Template 2D features of shape (batch, n_templates, nres, nres, 44)
            containing [distance_onehot(37), sin/cos_orientations(6), mask(1)]
    """
    B, T, L = xyz_t.shape[:3]
    c6d, mask = xyz_to_c6d(xyz_t[:,:,:,:3].view(B*T,L,3,3), params=params)
    c6d = c6d.view(B, T, L, L, 4)
    mask = mask.view(B, T, L, L, 1)
    #
    # dist to one-hot encoded
    dist = dist_to_onehot(c6d[...,0], params)
    orien = torch.cat((torch.sin(c6d[...,1:]), torch.cos(c6d[...,1:])), dim=-1)*mask # (B, T, L, L, 6)
    #
    mask = ~torch.isnan(c6d[:,:,:,:,0]) # (B, T, L, L)
    t2d = torch.cat((dist, orien, mask.unsqueeze(-1)), dim=-1)
    t2d[torch.isnan(t2d)] = 0.0
    return t2d

def xyz_to_chi1(xyz_t):
    """
    Extract chi1 sidechain torsion angles from template coordinates.

    Chi1 is the first sidechain dihedral angle, defined by atoms N-CA-CB-CG.
    It describes the rotational state of the sidechain about the CA-CB bond.
    Missing atoms (represented as NaN) are handled by setting the mask to 0.

    Args:
        xyz_t (torch.Tensor): Template atom coordinates of shape
            (batch, n_templates, nres, 14, 3)
            Atom order: [N, CA, C, O, CB, CG, ...]. Missing atoms should be NaN

    Returns:
        torch.Tensor: Chi1 features of shape (batch, n_templates, nres, 3)
            containing [cos(chi1), sin(chi1), mask] where mask is 1 if chi1
            is defined (all 4 atoms present) and 0 otherwise
    """
    B, T, L = xyz_t.shape[:3]
    xyz_t = xyz_t.reshape(B*T, L, 14, 3)
        
    # chi1 angle: N, CA, CB, CG
    chi1 = get_dih(xyz_t[:,:,0], xyz_t[:,:,1], xyz_t[:,:,4], xyz_t[:,:,5]) # (B*T, L)
    cos_chi1 = torch.cos(chi1)
    sin_chi1 = torch.sin(chi1)
    mask_chi1 = ~torch.isnan(chi1)
    chi1 = torch.stack((cos_chi1, sin_chi1, mask_chi1), dim=-1) # (B*T, L, 3)
    chi1[torch.isnan(chi1)] = 0.0
    chi1 = chi1.reshape(B, T, L, 3)
    return chi1

def xyz_to_bbtor(xyz, params=PARAMS):
    """
    Extract backbone torsion angles (phi, psi) from coordinates.

    Computes the two main backbone dihedral angles:
    - phi: Dihedral C(-1) - N - CA - C
    - psi: Dihedral N - CA - C - N(+1)

    These angles define the backbone conformation and are binned into
    discrete values for use as features.

    Args:
        xyz (torch.Tensor): Backbone coordinates of shape (batch, nres, 3, 3)
        params (dict, optional): Parameters including ABINS for angle binning

    Returns:
        torch.Tensor: Binned torsion angles of shape (batch, nres, 2) containing
            [phi_bin, psi_bin] as integer indices
    """
    batch = xyz.shape[0]
    nres = xyz.shape[1]

    # three anchor atoms
    N  = xyz[:,:,0]
    Ca = xyz[:,:,1]
    C  = xyz[:,:,2]

    # recreate Cb given N,Ca,C
    next_N = torch.roll(N, -1, dims=1)
    prev_C = torch.roll(C, 1, dims=1)
    phi = get_dih(prev_C, N, Ca, C)
    psi = get_dih(N, Ca, C, next_N)
    #
    phi[:,0] = 0.0
    psi[:,-1] = 0.0
    #
    astep = 2.0*np.pi / params['ABINS']
    phi_bin = torch.round((phi+np.pi-astep/2)/astep)
    psi_bin = torch.round((psi+np.pi-astep/2)/astep)
    return torch.stack([phi_bin, psi_bin], axis=-1).long()

# ============================================================
def dist_to_onehot(dist, params=PARAMS):
    """
    Convert continuous distances to one-hot encoded bins.

    Discretizes distance values into bins and creates a one-hot encoding.
    Distances outside the range [DMIN, DMAX] or NaN values are assigned
    to the last bin (bin index DBINS).

    Args:
        dist (torch.Tensor): Distance values of any shape
        params (dict, optional): Parameters with DMIN, DMAX, DBINS

    Returns:
        torch.Tensor: One-hot encoded distances of shape (*dist.shape, DBINS+1)
    """
    dist[torch.isnan(dist)] = 999.9
    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    dbins = torch.linspace(params['DMIN']+dstep, params['DMAX'], params['DBINS'],dtype=dist.dtype,device=dist.device)
    db = torch.bucketize(dist.contiguous(),dbins).long()
    dist = torch.nn.functional.one_hot(db, num_classes=params['DBINS']+1).float()
    return dist

def c6d_to_bins(c6d, params=PARAMS):
    """
    Bin 6D coordinates into discrete indices.

    Converts continuous 6D coordinate values (distance, omega, theta, phi)
    into discrete bin indices for use in classification or histogram-based
    representations.

    Args:
        c6d (torch.Tensor): 6D coordinates of shape (..., 4) containing
            [distance, omega, theta, phi]
        params (dict, optional): Binning parameters (DMIN, DMAX, DBINS, ABINS)

    Returns:
        torch.Tensor: Binned indices of shape (..., 4) as uint8 containing
            [dist_bin, omega_bin, theta_bin, phi_bin]
    """

    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    astep = 2.0*np.pi / params['ABINS']

    dbins = torch.linspace(params['DMIN']+dstep, params['DMAX'], params['DBINS'],dtype=c6d.dtype,device=c6d.device)
    ab360 = torch.linspace(-np.pi+astep, np.pi, params['ABINS'],dtype=c6d.dtype,device=c6d.device)
    ab180 = torch.linspace(astep, np.pi, params['ABINS']//2,dtype=c6d.dtype,device=c6d.device)

    db = torch.bucketize(c6d[...,0].contiguous(),dbins)
    ob = torch.bucketize(c6d[...,1].contiguous(),ab360)
    tb = torch.bucketize(c6d[...,2].contiguous(),ab360)
    pb = torch.bucketize(c6d[...,3].contiguous(),ab180)

    ob[db==params['DBINS']] = params['ABINS']
    tb[db==params['DBINS']] = params['ABINS']
    pb[db==params['DBINS']] = params['ABINS']//2

    return torch.stack([db,ob,tb,pb],axis=-1).to(torch.uint8)


# ============================================================
def dist_to_bins(dist, params=PARAMS):
    """
    Bin distance values into discrete indices.

    Args:
        dist (torch.Tensor): Distance values
        params (dict, optional): Parameters with DMIN, DMAX, DBINS

    Returns:
        torch.Tensor: Binned distance indices as long integers
    """

    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    db = torch.round((dist-params['DMIN']-dstep/2)/dstep)

    db[db<0] = 0
    db[db>params['DBINS']] = params['DBINS']
    
    return db.long()


# ============================================================
def c6d_to_bins2(c6d, same_chain, negative=False, params=PARAMS):
    """
    Bin 6D coordinates with support for multi-chain masking.

    Similar to c6d_to_bins but with additional logic for handling multi-chain
    complexes. When negative=True, inter-chain pairs are assigned to the
    no-contact bin.

    Args:
        c6d (torch.Tensor): 6D coordinates of shape (..., 4)
        same_chain (torch.Tensor): Boolean mask indicating same-chain pairs
        negative (bool, optional): If True, set inter-chain pairs to no-contact bin
        params (dict, optional): Binning parameters

    Returns:
        torch.Tensor: Binned indices of shape (..., 4) as long integers
    """

    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    astep = 2.0*np.pi / params['ABINS']

    db = torch.round((c6d[...,0]-params['DMIN']-dstep/2)/dstep)
    ob = torch.round((c6d[...,1]+np.pi-astep/2)/astep)
    tb = torch.round((c6d[...,2]+np.pi-astep/2)/astep)
    pb = torch.round((c6d[...,3]-astep/2)/astep)

    # put all d<dmin into one bin
    db[db<0] = 0
    
    # synchronize no-contact bins
    db[db>params['DBINS']] = params['DBINS']
    ob[db==params['DBINS']] = params['ABINS']
    tb[db==params['DBINS']] = params['ABINS']
    pb[db==params['DBINS']] = params['ABINS']//2
    
    if negative:
        db = torch.where(same_chain.bool(), db.long(), params['DBINS'])
        ob = torch.where(same_chain.bool(), ob.long(), params['ABINS'])
        tb = torch.where(same_chain.bool(), tb.long(), params['ABINS'])
        pb = torch.where(same_chain.bool(), pb.long(), params['ABINS']//2)
    
    return torch.stack([db,ob,tb,pb],axis=-1).long()

def get_init_xyz(xyz_t):
    """
    Initialize coordinates for missing residues using template information.

    When template coordinates are missing or incomplete, this function provides
    reasonable initial coordinates by:
    1. Using template coordinates where available
    2. For missing residues, copying coordinates from the nearest residue with
       coordinates and offsetting by the CA position

    This is useful for initializing structures before refinement.

    Args:
        xyz_t (torch.Tensor): Template coordinates of shape (batch, n_templates,
            nres, 14, 3). Missing atoms/residues are NaN

    Returns:
        torch.Tensor: Initialized coordinates of shape (batch, n_templates, nres,
            27, 3) with missing positions filled in
    """
    # input: xyz_t (B, T, L, 14, 3)
    # ouput: xyz (B, T, L, 14, 3)
    B, T, L = xyz_t.shape[:3]
    init = INIT_CRDS.to(xyz_t.device).reshape(1,1,1,27,3).repeat(B,T,L,1,1)
    if torch.isnan(xyz_t).all():
        return init

    mask = torch.isnan(xyz_t[:,:,:,:3]).any(dim=-1).any(dim=-1) # (B, T, L)
    #
    center_CA = ((~mask[:,:,:,None]) * torch.nan_to_num(xyz_t[:,:,:,1,:])).sum(dim=2) / ((~mask[:,:,:,None]).sum(dim=2)+1e-4) # (B, T, 3)
    xyz_t = xyz_t - center_CA.view(B,T,1,1,3)
    #
    idx_s = list()
    for i_b in range(B):
        for i_T in range(T):
            if mask[i_b, i_T].all():
                continue
            exist_in_templ = torch.where(~mask[i_b, i_T])[0] # (L_sub)
            seqmap = (torch.arange(L, device=xyz_t.device)[:,None] - exist_in_templ[None,:]).abs() # (L, L_sub)
            seqmap = torch.argmin(seqmap, dim=-1) # (L)
            idx = torch.gather(exist_in_templ, -1, seqmap) # (L)
            offset_CA = torch.gather(xyz_t[i_b, i_T, :, 1, :], 0, idx.reshape(L,1).expand(-1,3))
            init[i_b,i_T] += offset_CA.reshape(L,1,3)
    #
    xyz = torch.where(mask.view(B, T, L, 1, 1), init, xyz_t)
    return xyz
