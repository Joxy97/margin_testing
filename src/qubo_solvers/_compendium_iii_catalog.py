"""Torch-free, document-ordered IDs for Compendium III's 86 models."""

COMPENDIUM_III_SOLVERS = tuple('lib_c3_' + name for name in (
    'dzyaloshinskii_moriya', 'resonant_dipole_exchange', 'lipkin_meshkov_glick',
    'rydberg_pxp', 'aubry_andre', 'haldane', 'long_range_transverse_ising',
    'frustrated_j1_j2', 'bilinear_biquadratic', 'kitaev_heisenberg_gamma',
    'harper_hofstadter', 'holstein', 'rydberg_dressed_ising', 'bose_fermi_mixture',
    'degenerate_parametric_amplifier', 'trapped_ion_spin_phonon', 'molmer_sorensen',
    'kanamori', 'anderson_impurity', 'pariser_parr_pople', 'scalar_spin_chirality',
    'ring_exchange', 'central_spin', 'frohlich_polaron', 'lieb_liniger',
    'bosonic_josephson', 'spin_dependent_optical_lattice', 'optical_clock', 'dipolar_xxz',
    'kugel_khomskii', 'kondo_lattice', 'hubbard_holstein', 'pair_hopping_bose_hubbard',
    'quantum_east', 'quantum_p_spin', 'kicked_ising', 'wen_plaquette', 'schwinger',
    'xyzz_hexagonal_code', 'plaquette_ising', 'spin_transfer_torque', 'haah_cubic_code',
    'quantum_double', 'levin_wen', 'double_semion', 'quantum_link', 'majumdar_ghosh',
    'creutz_ladder', 'penson_kolb', 'richardson_gaudin', 'spinor_rydberg',
    'quantum_hopfield', 'quantum_potts', 'frustrated_spin_glass', 'blume_capel',
    'ashkin_teller', 'blume_emery_griffiths', 'fano_anderson', 'feshbach_two_channel',
    'optical_lattice', 'dirac_coulomb_breit', 'color_code', 'toric_code', 'surface_code',
    'xyz_spin', 'xzzx_surface_code', 'xx_spin', 'xxz_spin', 'kitaev_honeycomb',
    'quantum_compass', 'bacon_shor', 'rydberg_atom', 'rice_mele', 'rashba_bdg',
    'spin_orbit_bose_gas', 'raman_spin_orbit', 'tavis_cummings', 'lambda_three_level',
    'quantum_rabi', 'weyl_spin_orbit', 'rashba_zeeman', 'persistent_spin_helix',
    'rashba_dresselhaus', 'cubic_dresselhaus', 'dresselhaus', 'rashba',
))

COMPENDIUM_III_STRUCTURED_SOLVERS = tuple('lib_c3_' + name for name in (
    'xyzz_hexagonal_code', 'haah_cubic_code', 'quantum_double', 'color_code',
    'toric_code', 'surface_code', 'xzzx_surface_code', 'bacon_shor',
))
COMPENDIUM_III_QUBO_SOLVERS = tuple(
    name for name in COMPENDIUM_III_SOLVERS if name not in COMPENDIUM_III_STRUCTURED_SOLVERS
)
