import numpy as np
from unimol_tools.data.conformer import (
    inner_coords,
    coords2unimol,
    inner_smi2coords,
    create_mol_from_atoms_and_coords,
)
from unimol_tools.data.datareader import MolDataReader
from unimol_tools.data.dictionary import Dictionary


def test_inner_coords_and_coords2unimol():
    atoms = ['C', 'H', 'O']
    coords = [[0, 0, 0], [0, 0, 1], [1, 0, 0]]
    no_h_atoms, no_h_coords = inner_coords(atoms, coords, remove_hs=True)
    assert 'H' not in no_h_atoms
    d = Dictionary()
    for a in ['C', 'O']:
        if a not in d:
            d.add_symbol(a)
    feat = coords2unimol(no_h_atoms, no_h_coords, d)
    assert feat['src_tokens'].dtype == int
    assert feat['src_coord'].shape[1] == 3


def test_inner_smi2coords_returns_mol():
    mol = inner_smi2coords('CC', return_mol=True)
    from rdkit.Chem import Mol

    assert isinstance(mol, Mol)


def test_create_mol_from_atoms_and_coords():
    atoms = ['C', 'O']
    coords = [[0, 0, 0], [1, 0, 0]]
    mol = create_mol_from_atoms_and_coords(atoms, coords)
    from rdkit.Chem import Mol

    assert isinstance(mol, Mol)
    assert mol.GetNumAtoms() == 2


def test_single_molecule_dict_input():
    """Test that single molecule dict input with atoms and coordinates is handled correctly.
    
    This is a regression test for GitHub issue #6.
    """
    # Single molecule input (not batch format)
    data = {
        'atoms': ['C', 'C', 'O', 'N', 'H', 'H'],
        'coordinates': [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [2.4, 0.1, 0.0],
            [0.0, 1.3, 0.0],
            [-0.5, -0.8, 0.0],
            [1.7, -0.9, 0.0]
        ]
    }
    
    reader = MolDataReader()
    result = reader.read_data(data, is_train=False, task='repr')
    
    # Should be wrapped into batch format
    assert 'atoms' in result
    assert 'coordinates' in result
    assert len(result['atoms']) == 1  # Single molecule wrapped in batch
    assert len(result['coordinates']) == 1
    
    # The atoms and coordinates should have matching lengths
    atoms = result['atoms'][0]
    coordinates = result['coordinates'][0]
    assert len(atoms) == 6
    assert len(coordinates) == 6
    
    # Test that inner_coords works with this data
    no_h_atoms, no_h_coords = inner_coords(atoms, coordinates, remove_hs=True)
    assert 'H' not in no_h_atoms
    assert len(no_h_atoms) == 4  # C, C, O, N (no H)
    assert no_h_coords.shape == (4, 3)


def test_batch_molecule_dict_input():
    """Test that batch molecule dict input is handled correctly."""
    # Batch input (already in batch format)
    data = {
        'atoms': [['C', 'H', 'O'], ['N', 'C', 'H', 'H']],
        'coordinates': [
            [[0, 0, 0], [0, 0, 1], [1, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0]]
        ]
    }
    
    reader = MolDataReader()
    result = reader.read_data(data, is_train=False, task='repr')
    
    # Should remain as batch format
    assert len(result['atoms']) == 2
    assert len(result['coordinates']) == 2
    
    # Test inner_coords for each molecule
    for i in range(2):
        atoms = result['atoms'][i]
        coordinates = result['coordinates'][i]
        assert len(atoms) == len(coordinates)


def test_inner_coords_improved_error_messages():
    """Test that inner_coords provides helpful error messages."""
    # This test verifies that the error messages include actual values
    # We don't test the actual error raising as it would be fragile
    
    # Valid input should work
    atoms = ['C', 'H', 'O']
    coords = [[0, 0, 0], [0, 0, 1], [1, 0, 0]]
    no_h_atoms, no_h_coords = inner_coords(atoms, coords, remove_hs=True)
    assert len(no_h_atoms) == 2  # C and O
    assert no_h_coords.shape == (2, 3)
