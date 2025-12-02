"""
Potential energy management for guided diffusion.

This module manages the application of guiding potential functions during
diffusion sampling. It handles contact matrices for symmetric assemblies,
potential initialization, and gradient computation for structure guidance.
"""
import torch
from rfdiffusion.potentials import potentials as potentials
import numpy as np


def make_contact_matrix(nchain, intra_all=False, inter_all=False, contact_string=None):
    """
    Create a contact matrix for inter/intra-chain interactions.

    Builds a matrix indicating which chain pairs should have attractive (1),
    repulsive (-1), or no (0) contact potentials in symmetric oligomers.

    Args:
        nchain (int): Number of chains in the design
        intra_all (bool): Apply contact potential to all intra-chain pairs
        inter_all (bool): Apply contact potential to all inter-chain pairs
        contact_string (str, optional): Custom contact specification (e.g., 'A&B,B!C')
            '&' denotes attractive, '!' denotes repulsive

    Returns:
        np.ndarray: Contact matrix [nchain, nchain] with values -1, 0, or 1
    """
    alphabet   = [a for a in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    letter2num = {a:i for i,a in enumerate(alphabet)}
    
    contacts   = np.zeros((nchain,nchain))
    written    = np.zeros((nchain,nchain))
    
    
    # intra_all - everything on the diagonal has contact potential
    if intra_all:
        contacts[np.arange(nchain),np.arange(nchain)] = 1
    
    # inter all - everything off the diagonal has contact potential
    if inter_all:
        mask2d = np.full_like(contacts,False)
        for i in range(len(contacts)):
            for j in range(len(contacts)):
                if i!=j:
                    mask2d[i,j] = True
        
        contacts[mask2d.astype(bool)] = 1


    # custom contacts/repulsions from user 
    if contact_string != None:
        contact_list = contact_string.split(',') 
        for c in contact_list:
            assert len(c) == 3
            i,j = letter2num[c[0]],letter2num[c[2]]

            symbol = c[1]

            assert symbol in ['!','&']
            if symbol == '!':
                contacts[i,j] = -1
                contacts[j,i] = -1
            else:
                contacts[i,j] = 1
                contacts[j,i] = 1
            
    return contacts 


def calc_nchains(symbol, components=1):
    """
    Calculate total number of chains for a given symmetry.

    Args:
        symbol (str): Symmetry symbol (e.g., 'C3', 'D2', 'T')
        components (int): Number of asymmetric units

    Returns:
        int: Total number of chains in the symmetric assembly
    """
    S = symbol.lower()

    if S.startswith('c'):
        return int(S[1:])*components 
    elif S.startswith('d'):
        return 2*int(S[1:])*components 
    elif S.startswith('o'):
        raise NotImplementedError()
    elif S.startswith('t'):
        return 12*components
    else:
        raise RuntimeError('Unknown symmetry symbol ',S)


class PotentialManager:
    """
    Manager for guiding potential functions during diffusion sampling.

    This class initializes and applies various potential energy functions that
    guide the diffusion process toward desired structural properties. It handles
    potential scaling, decay schedules, and gradient computation for structure
    optimization.

    Supports potentials for:
    - Radius of gyration (compactness)
    - Interface/binder contacts
    - Substrate/ligand interactions
    - Symmetric oligomer contacts

    Author: NRB
    """

    def __init__(self,
                 potentials_config,
                 ppi_config,
                 diffuser_config,
                 inference_config,
                 hotspot_0idx,
                 binderlen,
                 ):
        """
        Initialize potential manager.

        Args:
            potentials_config: Configuration for potential functions
            ppi_config: Protein-protein interaction configuration
            diffuser_config: Diffusion process configuration
            inference_config: Inference settings including symmetry
            hotspot_0idx (list): 0-indexed hotspot residue positions
            binderlen (int): Length of binder chain
        """

        self.potentials_config = potentials_config
        self.ppi_config        = ppi_config
        self.inference_config  = inference_config

        self.guide_scale = potentials_config.guide_scale
        self.guide_decay = potentials_config.guide_decay
    
        if potentials_config.guiding_potentials is None: 
            setting_list = []
        else: 
            setting_list = [self.parse_potential_string(potstr) for potstr in potentials_config.guiding_potentials]


        # PPI potentials require knowledge about the binderlen which may be detected at runtime
        # This is a mechanism to still allow this info to be used in potentials - NRB 
        if binderlen > 0:
            binderlen_update   = { 'binderlen': binderlen }
            hotspot_res_update = { 'hotspot_res': hotspot_0idx }

            for setting in setting_list:
                if setting['type'] in potentials.require_binderlen:
                    setting.update(binderlen_update)

        self.potentials_to_apply = self.initialize_all_potentials(setting_list)
        self.T = diffuser_config.T
        
    def is_empty(self):
        """
        Check if any potentials are configured.

        Returns:
            bool: True if no potentials are active, False otherwise
        """

        return len(self.potentials_to_apply) == 0

    def parse_potential_string(self, potstr):
        """
        Parse potential configuration string to dictionary.

        Converts a string specification into a dictionary of potential settings.

        Args:
            potstr (str): Configuration string (e.g., 'type:binder_ROG,weight:1.0')

        Returns:
            dict: Parsed settings dictionary

        Example:
            'type:binder_ROG,weight:1.0,min_dist:15' ->
            {'type': 'binder_ROG', 'weight': 1.0, 'min_dist': 15.0}
        """

        setting_dict = {entry.split(':')[0]:entry.split(':')[1] for entry in potstr.split(',')}

        for key in setting_dict:
            if not key == 'type': setting_dict[key] = float(setting_dict[key])

        return setting_dict

    def initialize_all_potentials(self, setting_list):
        """
        Initialize all potential functions from configuration.

        Args:
            setting_list (list): List of potential configuration dictionaries

        Returns:
            list: List of initialized potential objects
        """

        to_apply = []

        for potential_dict in setting_list:
            assert(potential_dict['type'] in potentials.implemented_potentials), f'potential with name: {potential_dict["type"]} is not one of the implemented potentials: {potentials.implemented_potentials.keys()}'

            kwargs = {k: potential_dict[k] for k in potential_dict.keys() - {'type'}}

            # symmetric oligomer contact potential args
            if self.inference_config.symmetry:

                num_chains = calc_nchains(symbol=self.inference_config.symmetry, components=1) # hard code 1 for now 
                contact_kwargs={'nchain':num_chains,
                                'intra_all':self.potentials_config.olig_intra_all,
                                'inter_all':self.potentials_config.olig_inter_all,
                                'contact_string':self.potentials_config.olig_custom_contact }
                contact_matrix = make_contact_matrix(**contact_kwargs)
                kwargs.update({'contact_matrix':contact_matrix})


            to_apply.append(potentials.implemented_potentials[potential_dict['type']](**kwargs))

        return to_apply

    def compute_all_potentials(self, xyz):
        """
        Compute the sum of all active potential functions.

        This is the main function called during sampling to evaluate guiding potentials.

        Args:
            xyz (torch.Tensor): Current coordinates [L, 27, 3]

        Returns:
            torch.Tensor: Total potential energy (scalar to be maximized)
        """

        potential_list = [potential.compute(xyz) for potential in self.potentials_to_apply]
        potential_stack = torch.stack(potential_list, dim=0)

        return torch.sum(potential_stack, dim=0)

    def get_guide_scale(self, t):
        """
        Get the scale factor for guiding potentials at a given timestep.

        The scale factor can follow different schedules (constant, linear, quadratic, cubic)
        to control how strongly potentials influence the diffusion process over time.

        Args:
            t (int): Current timestep

        Returns:
            float: Scale factor for applying potential gradients
        """
        
        implemented_decay_types = {
                'constant': lambda t: self.guide_scale,
                # Linear interpolation with y2: 0, y1: guide_scale, x2: 0, x1: T, x: t
                'linear'  : lambda t: t/self.T * self.guide_scale,
                'quadratic' : lambda t: t**2/self.T**2 * self.guide_scale,
                'cubic' : lambda t: t**3/self.T**3 * self.guide_scale
        }
        
        if self.guide_decay not in implemented_decay_types:
            sys.exit(f'decay_type must be one of {implemented_decay_types.keys()}. Received decay_type={self.guide_decay}. Exiting.')
        
        return implemented_decay_types[self.guide_decay](t)


        
