"""
Symmetry handling for generating symmetric protein assemblies.

This module provides classes and functions for applying various types of molecular
symmetry (cyclic, dihedral, tetrahedral, octahedral, icosahedral) to protein
structures during generation. It handles coordinate transformations and chain
indexing for symmetric oligomers.
"""
from pyrsistent import v
from scipy.spatial.transform import Rotation
import functools as fn
import torch
import string
import logging
import numpy as np
import pathlib

format_rots = lambda r: torch.tensor(r).float()

T3_ROTATIONS = [
    torch.Tensor([
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0.,  1.]]).float(),
    torch.Tensor([
        [-1., -0.,  0.],
        [-0.,  1.,  0.],
        [-0.,  0., -1.]]).float(),
    torch.Tensor([
        [-1.,  0.,  0.],
        [ 0., -1.,  0.],
        [ 0.,  0.,  1.]]).float(),
    torch.Tensor([
        [ 1.,  0.,  0.],
        [ 0., -1.,  0.],
        [ 0.,  0., -1.]]).float(),
]

saved_symmetries = ['tetrahedral', 'octahedral', 'icosahedral']

class SymGen:
    """
    Symmetry generator for creating symmetric protein assemblies.

    This class handles the application of molecular symmetry operations to generate
    symmetric oligomers. Supports various point group symmetries including cyclic (Cn),
    dihedral (Dn), tetrahedral, octahedral, and icosahedral symmetries.
    """

    def __init__(self, global_sym, recenter, radius, model_only_neighbors=False):
        """
        Initialize symmetry generator.

        Args:
            global_sym (str): Symmetry type (e.g., 'C3', 'D2', 'octahedral')
            recenter (bool): Whether to recenter subunits after symmetry application
            radius (float): Radius for positioning subunits in certain symmetries
            model_only_neighbors (bool): Whether to model only neighboring subunits
        """
        self._log = logging.getLogger(__name__)
        self._recenter = recenter
        self._radius = radius

        if global_sym.lower().startswith('c'):
            # Cyclic symmetry
            if not global_sym[1:].isdigit():
                raise ValueError(f'Invalid cyclic symmetry {global_sym}')
            self._log.info(
                f'Initializing cyclic symmetry order {global_sym[1:]}.')
            self._init_cyclic(int(global_sym[1:]))
            self.apply_symmetry = self._apply_cyclic

        elif global_sym.lower().startswith('d'):
            # Dihedral symmetry
            if not global_sym[1:].isdigit():
                raise ValueError(f'Invalid dihedral symmetry {global_sym}')
            self._log.info(
                f'Initializing dihedral symmetry order {global_sym[1:]}.')
            self._init_dihedral(int(global_sym[1:]))
            # Applied the same way as cyclic symmetry
            self.apply_symmetry = self._apply_cyclic

        elif global_sym.lower() == 't3':
            # Tetrahedral (T3) symmetry
            self._log.info('Initializing T3 symmetry order.')
            self.sym_rots = T3_ROTATIONS
            self.order = 4
            # Applied the same way as cyclic symmetry
            self.apply_symmetry = self._apply_cyclic

        elif global_sym == 'octahedral':
            # Octahedral symmetry
            self._log.info(
                'Initializing octahedral symmetry.')
            self._init_octahedral()
            self.apply_symmetry = self._apply_octahedral

        elif global_sym.lower() in saved_symmetries:
            # Using a saved symmetry 
            self._log.info('Initializing %s symmetry order.'%global_sym)
            self._init_from_symrots_file(global_sym)

            # Applied the same way as cyclic symmetry
            self.apply_symmetry = self._apply_cyclic
        else:
            raise ValueError(f'Unrecognized symmetry {global_sym}')

        self.res_idx_procesing = fn.partial(
            self._lin_chainbreaks, num_breaks=self.order)

    #####################
    ## Cyclic symmetry ##
    #####################
    def _init_cyclic(self, order):
        """
        Initialize cyclic (Cn) symmetry.

        Creates rotation matrices for Cn symmetry by rotating around the z-axis
        by multiples of 360/n degrees.

        Args:
            order (int): Order of cyclic symmetry (n in Cn)
        """
        sym_rots = []
        for i in range(order):
            deg = i * 360.0 / order
            r = Rotation.from_euler('z', deg, degrees=True)
            sym_rots.append(format_rots(r.as_matrix()))
        self.sym_rots = sym_rots
        self.order = order

    def _apply_cyclic(self, coords_in, seq_in):
        """
        Apply cyclic symmetry to coordinates and sequence.

        Args:
            coords_in (torch.Tensor): Input coordinates [L, n_atoms, 3]
            seq_in (torch.Tensor): Input sequence [L, ...]

        Returns:
            tuple: (coords_out, seq_out) - Symmetrized coordinates and sequence
        """
        coords_out = torch.clone(coords_in)
        seq_out = torch.clone(seq_in)
        if seq_out.shape[0] % self.order != 0:
            raise ValueError(
                f'Sequence length must be divisble by {self.order}')
        subunit_len = seq_out.shape[0] // self.order
        for i in range(self.order):
            start_i = subunit_len * i
            end_i = subunit_len * (i+1)
            coords_out[start_i:end_i] = torch.einsum(
                'bnj,kj->bnk', coords_out[:subunit_len], self.sym_rots[i])
            seq_out[start_i:end_i]  = seq_out[:subunit_len]
        return coords_out, seq_out

    def _lin_chainbreaks(self, num_breaks, res_idx, offset=None):
        """
        Insert linear chain breaks for symmetric copies.

        Assigns different chain IDs and offsets residue indices for each
        symmetric subunit to distinguish them.

        Args:
            num_breaks (int): Number of chain breaks (= number of subunits)
            res_idx (torch.Tensor): Residue indices [B, L]
            offset (int, optional): Offset to add between chains

        Returns:
            tuple: (res_idx, chain_delimiters) - Updated indices and chain IDs
        """
        assert res_idx.ndim == 2
        res_idx = torch.clone(res_idx)
        subunit_len = res_idx.shape[-1] // num_breaks
        chain_delimiters = []
        if offset is None:
            offset = res_idx.shape[-1]
        for i in range(num_breaks):
            start_i = subunit_len * i
            end_i = subunit_len * (i+1)
            chain_labels = list(string.ascii_uppercase) + [str(i+j) for i in
                    string.ascii_uppercase for j in string.ascii_uppercase]
            chain_delimiters.extend(
                [chain_labels[i] for _ in range(subunit_len)]
            )
            res_idx[:, start_i:end_i] = res_idx[:, start_i:end_i] + offset * (i+1)
        return res_idx, chain_delimiters

    #######################
    ## Dihedral symmetry ##
    #######################
    def _init_dihedral(self, order):
        """
        Initialize dihedral (Dn) symmetry.

        Creates rotation matrices for Dn symmetry by combining z-axis rotations
        with a 180-degree flip around the x-axis.

        Args:
            order (int): Order of dihedral symmetry (n in Dn, total 2n subunits)
        """
        sym_rots = []
        flip = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        for i in range(order):
            deg = i * 360.0 / order
            rot = Rotation.from_euler('z', deg, degrees=True).as_matrix()
            sym_rots.append(format_rots(rot))
            rot2 = flip @ rot
            sym_rots.append(format_rots(rot2))
        self.sym_rots = sym_rots
        self.order = order * 2

    #########################
    ## Octahedral symmetry ##
    #########################
    def _init_octahedral(self):
        """
        Initialize octahedral symmetry.

        Loads pre-computed rotation matrices for octahedral point group symmetry
        from a saved file.
        """
        sym_rots = np.load(f"{pathlib.Path(__file__).parent.resolve()}/sym_rots.npz")
        self.sym_rots = [
            torch.tensor(v_i, dtype=torch.float32)
            for v_i in sym_rots['octahedral']
        ]
        self.order = len(self.sym_rots)

    def _apply_octahedral(self, coords_in, seq_in):
        """
        Apply octahedral symmetry to coordinates and sequence.

        Applies rotation matrices and optionally recenters each subunit
        at a specified radius from the origin.

        Args:
            coords_in (torch.Tensor): Input coordinates [L, n_atoms, 3]
            seq_in (torch.Tensor): Input sequence [L, ...]

        Returns:
            tuple: (coords_out, seq_out) - Symmetrized coordinates and sequence
        """
        coords_out = torch.clone(coords_in)
        seq_out = torch.clone(seq_in)
        if seq_out.shape[0] % self.order != 0:
            raise ValueError(
                f'Sequence length must be divisble by {self.order}')
        subunit_len = seq_out.shape[0] // self.order
        base_axis = torch.tensor([self._radius, 0., 0.])[None]
        for i in range(self.order):
            start_i = subunit_len * i
            end_i = subunit_len * (i+1)
            subunit_chain = torch.einsum(
                'bnj,kj->bnk', coords_in[:subunit_len], self.sym_rots[i])

            if self._recenter:
                center = torch.mean(subunit_chain[:, 1, :], axis=0)
                subunit_chain -= center[None, None, :]
                rotated_axis = torch.einsum(
                    'nj,kj->nk', base_axis, self.sym_rots[i]) 
                subunit_chain += rotated_axis[:, None, :]

            coords_out[start_i:end_i] = subunit_chain
            seq_out[start_i:end_i]  = seq_out[:subunit_len]
        return coords_out, seq_out

    #######################
    ## symmetry from file #
    #######################
    def _init_from_symrots_file(self, name):
        """
        Initialize symmetry from pre-computed rotation matrices.

        Loads rotation matrices from ./inference/sym_rots.npz for complex
        symmetry types (tetrahedral, octahedral, icosahedral).

        Args:
            name (str): Symmetry name ('tetrahedral', 'octahedral', 'icosahedral')

        Sets:
            self.sym_rots: List of rotation matrices [3, 3]
            self.order: Number of symmetry operations
        """
        assert name in saved_symmetries, name + " not in " + str(saved_symmetries)

        # Load in list of rotation matrices for `name`
        fn = f"{pathlib.Path(__file__).parent.resolve()}/sym_rots.npz"
        obj = np.load(fn)
        symms = None
        for k, v in obj.items():
            if str(k) == name: symms = v
        assert symms is not None, "%s not found in %s"%(name, fn)

        
        self.sym_rots =  [torch.tensor(v_i, dtype=torch.float32) for v_i in symms]
        self.order = len(self.sym_rots)

        # Return if identity is the first rotation  
        if not np.isclose(((self.sym_rots[0]-np.eye(3))**2).sum(), 0):

            # Move identity to be the first rotation
            for i, rot in enumerate(self.sym_rots):
                if np.isclose(((rot-np.eye(3))**2).sum(), 0):
                    self.sym_rots = [self.sym_rots.pop(i)]  + self.sym_rots

            assert len(self.sym_rots) == self.order
            assert np.isclose(((self.sym_rots[0]-np.eye(3))**2).sum(), 0)

    def close_neighbors(self):
        """
        Find rotation matrices corresponding to close neighbor subunits.

        Identifies which symmetry operations produce subunits that are close
        neighbors in the symmetric assembly by finding rotations with small
        rotation angles.

        Returns:
            list: Rotation matrices for identity and close neighboring operations
        """
        # set of small rotation angle rotations
        rel_rot = lambda M: np.linalg.norm(Rotation.from_matrix(M).as_rotvec())
        rel_rots = [(i+1, rel_rot(M)) for i, M in enumerate(self.sym_rots[1:])]
        min_rot = min(rel_rot_val[1] for rel_rot_val in rel_rots)
        close_rots = [np.eye(3)] + [
                self.sym_rots[i] for i, rel_rot_val in rel_rots if
                np.isclose(rel_rot_val, min_rot)
                ]
        return close_rots
