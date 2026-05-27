"""Emission Line Lookup Table

Comprehensive table of UV/optical emission lines observed in galaxies,
AGN, and QSOs. Extracted from:
"Table of UV/optical Emission Lines Observed in Galaxies"
Source: http://astronomy.nmsu.edu/drewski/tableofemissionlines.html

Wavelengths are in Angstroms in AIR (for λ > 2000 Å) or VACUUM (for λ < 2000 Å)

Note: This table contains critical and strong lines for optical range (3000-11000 Å).
For the complete table with 180+ lines, refer to the website.
"""

# Critical emission lines (MUST protect during noise masking)
# Wavelengths verified against website: http://astronomy.nmsu.edu/drewski/tableofemissionlines.html
CRITICAL_LINES = {
    # Balmer series (H I) - MOST CRITICAL
    'H_ALPHA': 6562.819,  # H α
    'H_BETA': 4861.333,   # H β
    'H_GAMMA': 4340.471,  # H γ
    'H_DELTA': 4101.742,  # H δ
    'H_EPSILON': 3970.079, # H ε
    'H_ZETA': 3889.064,  # H ζ (H8)
    'H_ETA': 3835.391,   # H η (H9)
    'H_THETA': 3797.904,  # H θ (H10)

    # [O III] - CRITICAL
    'O_III_5007': 5006.843,  # [O III] 5007
    'O_III_4959': 4958.911,  # [O III] 4959
    'O_III_4363': 4363.210,  # [O III] 4363

    # [O II] - CRITICAL (you identified this!)
    'O_II_3727A': 3726.032,  # [O II] 3726
    'O_II_3727B': 3728.815,  # [O II] 3728
    'O_II_7320A': 7319.990,  # [O II] 7319
    'O_II_7320B': 7330.730,  # [O II] 7330

    # [N II] - CRITICAL
    'N_II_6583': 6583.460,   # [N II] 6583
    'N_II_6548': 6548.050,   # [N II] 6548
    'N_II_5754': 5754.590,   # [N II] 5754

    # [S II] - CRITICAL
    'S_II_6716': 6716.440,   # [S II] 6716
    'S_II_6731': 6730.810,   # [S II] 6731
    'S_II_4076': 4076.349,   # [S II] 4076
    'S_II_4068': 4068.600,   # [S II] 4068

    # [O I] - Important
    'O_I_6300': 6300.304,    # [O I] 6300
    'O_I_6363': 6363.776,    # [O I] 6363

    # Helium - Important
    'HE_II_4686': 4685.710,  # He II 4686
    'HE_I_5876': 5875.624,   # He I 5876
    'HE_I_4471': 4471.479,   # He I 4471
    'HE_I_3889': 3888.647,   # He I 3889

    # [Ne III] - Important
    'NE_III_3869': 3868.760, # [Ne III] 3868
    'NE_III_3967': 3967.470, # [Ne III] 3967

    # [Ar III] - Important
    'AR_III_7136': 7135.790, # [Ar III] 7136
    'AR_III_7751': 7751.060, # [Ar III] 7751

    # [S III] - Important
    'SIII_6312': 6312.060,   # [S III] 6312
    'SIII_9069': 9068.600,   # [S III] 9069

    # Silicon - Important
    'SI_II_6347': 6347.100,   # Si II 6347
}

# Additional strong lines (less critical but common)
STRONG_LINES = {
    # Carbon
    'C_III_1909': 1908.734,   # C III] 1909
    'C_III_977': 977.030,     # C III 977
    'C_IV_1548': 1548.187,     # C IV 1548
    'C_IV_1550': 1550.772,     # C IV 1550
    'C_II_1335': 1335.708,     # C II 1335
    'C_II_2324': 2324.690,     # C II] 2324
    'C_III_5696': 5695.920,     # C III 5696 (WR feature)

    # Nitrogen
    'N_I_5200': 5200.257,      # [N I] 5200
    'N_III_1749': 1748.656,     # N III] 1749
    'N_V_1242': 1242.804,       # N V 1242
    'N_V_1238': 1238.821,       # N V 1238

    # Oxygen additional
    'O_I_5577': 5577.0,         # O I 5577
    'O_I_7002': 7002.230,       # O I 7002
    'O_I_8446': 8446.359,       # O I 8446

    # Neon
    'NE_V_3426': 3425.881,      # [Ne V] 3426
    'NE_V_3343': 3345.821,      # [Ne V] 3343

    # Magnesium
    'MG_II_2803': 2802.705,     # Mg II] 2803
    'MG_II_2796': 2795.528,     # Mg II] 2796

    # Silicon
    'SI_II_1260': 1260.422,      # Si II 1260
    'SI_III_1892': 1892.030,     # Si III] 1892

    # Sulfur additional
    'S_II_10287': 10286.730,    # [S II] 10287
    'S_II_10320': 10320.490,    # [S II] 10320
    'S_II_10336': 10336.410,    # [S II] 10336

    # Calcium (infrared Paschen region)
    'CA_II_8498': 8498.020,      # Ca II 8498
    'CA_II_8542': 8542.090,      # Ca II 8542
    'CA_II_8662': 8662.140,      # Ca II 8662

    # Iron
    'FE_II_4233': 4233.172,
    'FE_II_4924': 4923.927,
    'FE_II_5018': 5018.440,
    'FE_II_5169': 5169.033,
    'FE_II_5276': 5276.002,
    'FE_II_6369': 6369.462,
    'FE_II_6516': 6516.081,
    'FE_II_7155': 7155.157,
    'FE_II_7172': 7172.000,
    'FE_II_7452': 7452.538,
    'FE_II_8617': 8616.950,
    'FE_II_8892': 8891.910,
}

def get_protected_waves():
    """Return list of critical emission line wavelengths to protect"""
    return list(CRITICAL_LINES.values())

def get_all_strong_waves():
    """Return list of all strong emission line wavelengths"""
    waves = list(CRITICAL_LINES.values()) + list(STRONG_LINES.values())
    return sorted(set(waves))
