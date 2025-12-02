"""
Potential energy functions for guided protein structure generation.

This module implements various differentiable potential functions that guide
the diffusion process toward desired structural properties. Potentials include:
- Radius of gyration (compactness)
- Contact numbers (intra-protein, interface, oligomer)
- Substrate/ligand interactions

All potentials return values to be MAXIMIZED (negative for energies to minimize).
"""
import torch
import numpy as np
from rfdiffusion.util import generate_Cbeta

class Potential:
    """
    Base class interface for potential energy functions.

    All potential implementations must inherit from this class and implement
    the compute() method.
    """

    def compute(self, xyz):
        """
        Compute the potential energy for the current structure.

        Args:
            xyz (torch.Tensor): Current coordinates [L, 27, 3]

        Returns:
            torch.Tensor: Scalar potential value to be MAXIMIZED
                         (take gradient ascent steps)
        """
        raise NotImplementedError('Potential compute function was not overwritten')

class monomer_ROG(Potential):
    """
    Radius of gyration potential for monomer compactness.

    Encourages the structure to be compact by penalizing large radius of gyration.
    Distances below min_dist are not penalized to avoid over-compaction.

    Author: DJ (refactored by NRB)
    """

    def __init__(self, weight=1, min_dist=15):
        """
        Initialize monomer ROG potential.

        Args:
            weight (float): Scaling factor for the potential
            min_dist (float): Minimum distance from centroid (not penalized below this)
        """

        self.weight   = weight
        self.min_dist = min_dist

    def compute(self, xyz):
        Ca = xyz[:,1] # [L,3]

        centroid = torch.mean(Ca, dim=0, keepdim=True) # [1,3]

        dgram = torch.cdist(Ca[None,...].contiguous(), centroid[None,...].contiguous(), p=2) # [1,L,1,3]

        dgram = torch.maximum(self.min_dist * torch.ones_like(dgram.squeeze(0)), dgram.squeeze(0)) # [L,1,3]

        rad_of_gyration = torch.sqrt( torch.sum(torch.square(dgram)) / Ca.shape[0] ) # [1]

        return -1 * self.weight * rad_of_gyration

class binder_ROG(Potential):
    """
    Radius of gyration potential for binder chain compactness.

    Similar to monomer_ROG but only considers the binder portion of a complex.

    Author: NRB
    """

    def __init__(self, binderlen, weight=1, min_dist=15):
        """
        Initialize binder ROG potential.

        Args:
            binderlen (int): Length of the binder chain
            weight (float): Scaling factor
            min_dist (float): Minimum distance threshold
        """

        self.binderlen = binderlen
        self.min_dist  = min_dist
        self.weight    = weight

    def compute(self, xyz):
        
        # Only look at binder residues
        Ca = xyz[:self.binderlen,1] # [Lb,3]

        centroid = torch.mean(Ca, dim=0, keepdim=True) # [1,3]

        # cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), centroid[None,...].contiguous(), p=2) # [1,Lb,1,3]

        dgram = torch.maximum(self.min_dist * torch.ones_like(dgram.squeeze(0)), dgram.squeeze(0)) # [Lb,1,3]

        rad_of_gyration = torch.sqrt( torch.sum(torch.square(dgram)) / Ca.shape[0] ) # [1]

        return -1 * self.weight * rad_of_gyration


class dimer_ROG(Potential):
    """
    Radius of gyration potential for symmetric dimer compactness.

    Encourages both monomers in a dimer to be compact by averaging their
    individual ROG penalties.

    Author: PV
    """

    def __init__(self, binderlen, weight=1, min_dist=15):
        """
        Initialize dimer ROG potential.

        Args:
            binderlen (int): Length of each monomer
            weight (float): Scaling factor
            min_dist (float): Minimum distance threshold
        """

        self.binderlen = binderlen
        self.min_dist  = min_dist
        self.weight    = weight

    def compute(self, xyz):

        # Only look at monomer 1 residues
        Ca_m1 = xyz[:self.binderlen,1] # [Lb,3]
        
        # Only look at monomer 2 residues
        Ca_m2 = xyz[self.binderlen:,1] # [Lb,3]

        centroid_m1 = torch.mean(Ca_m1, dim=0, keepdim=True) # [1,3]
        centroid_m2 = torch.mean(Ca_m1, dim=0, keepdim=True) # [1,3]

        # cdist needs a batch dimension - NRB
        #This calculates RoG for Monomer 1
        dgram_m1 = torch.cdist(Ca_m1[None,...].contiguous(), centroid_m1[None,...].contiguous(), p=2) # [1,Lb,1,3]
        dgram_m1 = torch.maximum(self.min_dist * torch.ones_like(dgram_m1.squeeze(0)), dgram_m1.squeeze(0)) # [Lb,1,3]
        rad_of_gyration_m1 = torch.sqrt( torch.sum(torch.square(dgram_m1)) / Ca_m1.shape[0] ) # [1]

        # cdist needs a batch dimension - NRB
        #This calculates RoG for Monomer 2
        dgram_m2 = torch.cdist(Ca_m2[None,...].contiguous(), centroid_m2[None,...].contiguous(), p=2) # [1,Lb,1,3]
        dgram_m2 = torch.maximum(self.min_dist * torch.ones_like(dgram_m2.squeeze(0)), dgram_m2.squeeze(0)) # [Lb,1,3]
        rad_of_gyration_m2 = torch.sqrt( torch.sum(torch.square(dgram_m2)) / Ca_m2.shape[0] ) # [1]

        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return -1 * self.weight * (rad_of_gyration_m1 + rad_of_gyration_m2)/2

class binder_ncontacts(Potential):
    """
    Differentiable contact number potential for binder compactness.

    Uses a smooth switching function to count contacts, similar to PLUMED's
    coordination number. Encourages formation of contacts within the binder.

    Reference: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html
    """

    def __init__(self, binderlen, weight=1, r_0=8, d_0=4):
        """
        Initialize binder contact potential.

        Args:
            binderlen (int): Length of binder chain
            weight (float): Scaling factor
            r_0 (float): Distance scaling parameter
            d_0 (float): Distance offset parameter
        """

        self.binderlen = binderlen
        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0

    def compute(self, xyz):

        # Only look at binder Ca residues
        Ca = xyz[:self.binderlen,1] # [Lb,3]
        
        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        binder_ncontacts = (1 - numerator) / (1 - denominator)
        
        print("BINDER CONTACTS:", binder_ncontacts.sum())
        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return self.weight * binder_ncontacts.sum()

class interface_ncontacts(Potential):
    """
    Contact potential for binder-target interface.

    Encourages formation of contacts across the interface between binder
    and target using a differentiable coordination number.

    Reference: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html

    Author: PV
    """

    def __init__(self, binderlen, weight=1, r_0=8, d_0=6):
        """
        Initialize interface contact potential.

        Args:
            binderlen (int): Length of binder chain
            weight (float): Scaling factor
            r_0 (float): Distance scaling parameter
            d_0 (float): Distance offset parameter
        """

        self.binderlen = binderlen
        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0

    def compute(self, xyz):

        # Extract binder Ca residues
        Ca_b = xyz[:self.binderlen,1] # [Lb,3]

        # Extract target Ca residues
        Ca_t = xyz[self.binderlen:,1] # [Lt,3]

        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca_b[None,...].contiguous(), Ca_t[None,...].contiguous(), p=2) # [1,Lb,Lt]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        interface_ncontacts = (1 - numerator) / (1 - denominator)
        #Potential is the sum of values in the tensor
        interface_ncontacts = interface_ncontacts.sum()

        print("INTERFACE CONTACTS:", interface_ncontacts.sum())

        return self.weight * interface_ncontacts


class monomer_contacts(Potential):
    """
    Contact potential for monomer self-interactions.

    Encourages formation of intra-protein contacts using a differentiable
    coordination number for the entire protein.

    Reference: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html

    Author: PV

    Note: This function sometimes produces NaN gradients - there is a check
          in reverse diffusion to handle this.
    """

    def __init__(self, weight=1, r_0=8, d_0=2, eps=1e-6):
        """
        Initialize monomer contact potential.

        Args:
            weight (float): Scaling factor
            r_0 (float): Distance scaling parameter
            d_0 (float): Distance offset parameter
            eps (float): Small constant for numerical stability
        """

        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0
        self.eps       = eps

    def compute(self, xyz):

        Ca = xyz[:,1] # [L,3]

        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)

        ncontacts = (1 - numerator) / ((1 - denominator))


        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return self.weight * ncontacts.sum()


class olig_contacts(Potential):
    """
    Contact potential for symmetric oligomers with custom inter/intra-chain rules.

    Applies contact potentials within and between chains in symmetric assemblies.
    Can specify attractive/repulsive interactions between specific chain pairs.

    Author: DJ
    """

    def __init__(self,
                 contact_matrix,
                 weight_intra=1,
                 weight_inter=1,
                 r_0=8, d_0=2):
        """
        Initialize oligomer contact potential.

        Args:
            contact_matrix (np.ndarray): Square matrix [Nchains, Nchains] where
                entry (i,j) is 1 (attractive), -1 (repulsive), or 0 (no potential)
            weight_intra (float): Weight for intra-chain contacts
            weight_inter (float): Weight for inter-chain contacts
            r_0 (float): Distance scaling parameter
            d_0 (float): Distance offset parameter
        """
        self.contact_matrix = contact_matrix
        self.weight_intra = weight_intra 
        self.weight_inter = weight_inter 
        self.r_0 = r_0
        self.d_0 = d_0

        # check contact matrix only contains valid entries 
        assert all([i in [-1,0,1] for i in contact_matrix.flatten()]), 'Contact matrix must contain only 0, 1, or -1 in entries'
        # assert the matrix is square and symmetric 
        shape = contact_matrix.shape 
        assert len(shape) == 2 
        assert shape[0] == shape[1]
        for i in range(shape[0]):
            for j in range(shape[1]):
                assert contact_matrix[i,j] == contact_matrix[j,i]
        self.nchain=shape[0]

         
    def _get_idx(self,i,L):
        """
        Returns the zero-indexed indices of the residues in chain i
        """
        assert L%self.nchain == 0
        Lchain = L//self.nchain
        return i*Lchain + torch.arange(Lchain)


    def compute(self, xyz):
        """
        Iterate through the contact matrix, compute contact potentials between chains that need it,
        and negate contacts for any 
        """
        L = xyz.shape[0]

        all_contacts = 0
        start = 0
        for i in range(self.nchain):
            for j in range(self.nchain):
                # only compute for upper triangle, disregard zeros in contact matrix 
                if (i <= j) and (self.contact_matrix[i,j] != 0):

                    # get the indices for these two chains 
                    idx_i = self._get_idx(i,L)
                    idx_j = self._get_idx(j,L)

                    Ca_i = xyz[idx_i,1]  # slice out crds for this chain 
                    Ca_j = xyz[idx_j,1]  # slice out crds for that chain 
                    dgram           = torch.cdist(Ca_i[None,...].contiguous(), Ca_j[None,...].contiguous(), p=2) # [1,Lb,Lb]

                    divide_by_r_0   = (dgram - self.d_0) / self.r_0
                    numerator       = torch.pow(divide_by_r_0,6)
                    denominator     = torch.pow(divide_by_r_0,12)
                    ncontacts       = (1 - numerator) / (1 - denominator)

                    # weight, don't double count intra 
                    scalar = (i==j)*self.weight_intra/2 + (i!=j)*self.weight_inter

                    #                 contacts              attr/repuls          relative weights 
                    all_contacts += ncontacts.sum() * self.contact_matrix[i,j] * scalar 

        return all_contacts 
                    
def get_damped_lj(r_min, r_lin,p1=6,p2=12):
    
    y_at_r_lin = lj(r_lin, r_min, p1, p2)
    ydot_at_r_lin = lj_grad(r_lin, r_min,p1,p2)
    
    def inner(dgram):
        return (dgram < r_lin) * (ydot_at_r_lin * (dgram - r_lin) + y_at_r_lin) + (dgram >= r_lin) * lj(dgram, r_min, p1, p2)
    return inner

def lj(dgram, r_min,p1=6, p2=12):
    return 4 * ((r_min / (2**(1/p1) * dgram))**p2 - (r_min / (2**(1/p1) * dgram))**p1)

def lj_grad(dgram, r_min,p1=6,p2=12):
    return -p2 * r_min**p1*(r_min**p1-dgram**p1) / (dgram**(p2+1))

def mask_expand(mask, n=1):
    mask_out = mask.clone()
    assert mask.ndim == 1
    for i in torch.where(mask)[0]:
        for j in range(i-n, i+n+1):
            if j >= 0 and j < len(mask):
                mask_out[j] = True
    return mask_out

def contact_energy(dgram, d_0, r_0):
    """
    Compute smooth contact energy from distance matrix.

    Uses a switching function to smoothly count contacts.

    Args:
        dgram (torch.Tensor): Distance matrix
        d_0 (float): Distance offset
        r_0 (float): Distance scale

    Returns:
        torch.Tensor: Negative contact count (to maximize)
    """
    divide_by_r_0 = (dgram - d_0) / r_0
    numerator = torch.pow(divide_by_r_0,6)
    denominator = torch.pow(divide_by_r_0,12)
    
    ncontacts = (1 - numerator) / ((1 - denominator)).float()
    return - ncontacts

def poly_repulse(dgram, r, slope, p=1):
    """
    Polynomial repulsion potential for close contacts.

    Args:
        dgram (torch.Tensor): Distance matrix
        r (float): Cutoff distance
        slope (float): Steepness of repulsion
        p (float): Power for polynomial

    Returns:
        torch.Tensor: Repulsive energy
    """
    a = slope / (p * r**(p-1))

    return (dgram < r) * a * torch.abs(r - dgram)**p * slope

#def only_top_n(dgram


class substrate_contacts(Potential):
    """
    Potential for substrate/ligand binding site design.

    Implicitly models a ligand using attractive-repulsive potentials to guide
    the formation of binding sites. Uses affine transformations to track ligand
    position relative to motif residues.
    """

    def __init__(self, weight=1, r_0=8, d_0=2, s=1, eps=1e-6, rep_r_0=5, rep_s=2, rep_r_min=1):
        """
        Initialize substrate contact potential.

        Args:
            weight (float): Overall scaling factor
            r_0 (float): Distance scale for attractive potential
            d_0 (float): Distance offset for attractive potential
            s (float): Strength of attractive potential
            eps (float): Numerical stability constant
            rep_r_0 (float): Distance scale for repulsive potential
            rep_s (float): Strength of repulsive potential
            rep_r_min (float): Minimum distance for repulsion
        """

        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0
        self.eps       = eps
        
        # motif frame coordinates
        # NOTE: these probably need to be set after sample_init() call, because the motif sequence position in design must be known
        self.motif_frame = None # [4,3] xyz coordinates from 4 atoms of input motif
        self.motif_mapping = None # list of tuples giving positions of above atoms in design [(resi, atom_idx)]
        self.motif_substrate_atoms = None # xyz coordinates of substrate from input motif
        r_min = 2
        self.energies = []
        self.energies.append(lambda dgram: s * contact_energy(torch.min(dgram, dim=-1)[0], d_0, r_0))
        if rep_r_min:
            self.energies.append(lambda dgram: poly_repulse(torch.min(dgram, dim=-1)[0], rep_r_0, rep_s, p=1.5))
        else:
            self.energies.append(lambda dgram: poly_repulse(dgram, rep_r_0, rep_s, p=1.5))


    def compute(self, xyz):
        
        # First, get random set of atoms
        # This operates on self.xyz_motif, which is assigned to this class in the model runner (for horrible plumbing reasons)
        self._grab_motif_residues(self.xyz_motif)
        
        # for checking affine transformation is corect
        first_distance = torch.sqrt(torch.sqrt(torch.sum(torch.square(self.motif_substrate_atoms[0] - self.motif_frame[0]), dim=-1))) 

        # grab the coordinates of the corresponding atoms in the new frame using mapping
        res = torch.tensor([k[0] for k in self.motif_mapping])
        atoms = torch.tensor([k[1] for k in self.motif_mapping])
        new_frame = xyz[self.diffusion_mask][res,atoms,:]
        # calculate affine transformation matrix and translation vector b/w new frame and motif frame
        A, t = self._recover_affine(self.motif_frame, new_frame)
        # apply affine transformation to substrate atoms
        substrate_atoms = torch.mm(A, self.motif_substrate_atoms.transpose(0,1)).transpose(0,1) + t
        second_distance = torch.sqrt(torch.sqrt(torch.sum(torch.square(new_frame[0] - substrate_atoms[0]), dim=-1)))
        assert abs(first_distance - second_distance) < 0.01, "Alignment seems to be bad" 
        diffusion_mask = mask_expand(self.diffusion_mask, 1)
        Ca = xyz[~diffusion_mask, 1]

        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), substrate_atoms.float()[None], p=2)[0] # [Lb,Lb]

        all_energies = []
        for i, energy_fn in enumerate(self.energies):
            energy = energy_fn(dgram)
            all_energies.append(energy.sum())
        return - self.weight * sum(all_energies)

        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return self.weight * ncontacts.sum()

    def _recover_affine(self,frame1, frame2):
        """
        Uses Simplex Affine Matrix (SAM) formula to recover affine transform between two sets of 4 xyz coordinates
        See: https://www.researchgate.net/publication/332410209_Beginner%27s_guide_to_mapping_simplexes_affinely

        Args: 
        frame1 - 4 coordinates from starting frame [4,3]
        frame2 - 4 coordinates from ending frame [4,3]
        
        Outputs:
        A - affine transformation matrix from frame1->frame2
        t - affine translation vector from frame1->frame2
        """

        l = len(frame1)
        # construct SAM denominator matrix
        B = torch.vstack([frame1.T, torch.ones(l)])
        D = 1.0 / torch.linalg.det(B) # SAM denominator

        M = torch.zeros((3,4), dtype=torch.float64)
        for i, R in enumerate(frame2.T):
            for j in range(l):
                num = torch.vstack([R, B])
                # make SAM numerator matrix
                num = torch.cat((num[:j+1],num[j+2:])) # make numerator matrix
                # calculate SAM entry
                M[i][j] = (-1)**j * D * torch.linalg.det(num)

        A, t = torch.hsplit(M, [l-1])
        t = t.transpose(0,1)
        return A, t

    def _grab_motif_residues(self, xyz) -> None:
        """
        Grabs 4 atoms in the motif.
        Currently random subset of Ca atoms if the motif is >= 4 residues, or else 4 random atoms from a single residue
        """
        idx = torch.arange(self.diffusion_mask.shape[0])
        idx = idx[self.diffusion_mask].float()
        if torch.sum(self.diffusion_mask) >= 4:
            rand_idx = torch.multinomial(idx, 4).long()
            # get Ca atoms
            self.motif_frame = xyz[rand_idx, 1]
            self.motif_mapping = [(i,1) for i in rand_idx]
        else:
            rand_idx = torch.multinomial(idx, 1).long()
            self.motif_frame = xyz[rand_idx[0],:4]
            self.motif_mapping = [(rand_idx, i) for i in range(4)]

# Dictionary mapping potential names to classes. Used by PotentialManager.
# New potentials must be added here to be accessible via configuration strings.
implemented_potentials = { 'monomer_ROG':          monomer_ROG,
                           'binder_ROG':           binder_ROG,
                           'dimer_ROG':            dimer_ROG,
                           'binder_ncontacts':     binder_ncontacts,
                           'interface_ncontacts':  interface_ncontacts,
                           'monomer_contacts':     monomer_contacts,
                           'olig_contacts':        olig_contacts,
                           'substrate_contacts':    substrate_contacts}

# Set of potentials that require binderlen parameter
require_binderlen      = { 'binder_ROG',
                           'binder_distance_ReLU',
                           'binder_any_ReLU',
                           'dimer_ROG',
                           'binder_ncontacts',
                           'interface_ncontacts'}

