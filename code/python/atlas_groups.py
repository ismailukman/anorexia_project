"""
atlas_groups.py: functional groups of the combined 216-region atlas.

Single source for the group vector used by every combined-216 analysis.
Cortical regions take their Yeo-7 network from the Schaefer label. Each Tian S1
region takes the network assigned by 01e_subcortical_network_assignment.py
(winner-take-all on control functional connectivity) and forms a separate
subcortical group named "~<network>", so cortical and subcortical members of a
network are kept apart.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from paths import ROOT
LABELS = ROOT / "data/atlas/combined_216/combined_216_labels.txt"
ASSIGNMENT = ROOT / "data/atlas/combined_216/subcortical_network_assignment.csv"

YEO_KEYS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
YEO_NAMES = ["VN", "SMN", "DAN", "VAN", "LN", "FPN", "DMN"]


def labels216():
    return [l.split()[1] for l in LABELS.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


# "network" is the primary assignment; "network_alt" the runner-up, used for
# the sensitivity analysis (05b --groups alt).
MAPPING = "network"


def subcortical_networks(column=None):
    """Tian region label -> assigned Yeo-7 network abbreviation."""
    df = pd.read_csv(ASSIGNMENT)
    return dict(zip(df.region, df[column or MAPPING]))


def combined216_groups():
    """Return (group_vec 1..G, group_names, labels) for the 216 regions."""
    labels = labels216()
    assign = subcortical_networks()
    sub_nets = [n for n in YEO_NAMES if n in set(assign.values())]
    names = YEO_NAMES + ["~" + n for n in sub_nets]
    gv = np.zeros(len(labels), dtype=int)
    for i, lbl in enumerate(labels):
        if i < 200:
            gv[i] = next(k for k, key in enumerate(YEO_KEYS) if key in lbl) + 1
        else:
            gv[i] = len(YEO_NAMES) + sub_nets.index(assign[lbl]) + 1
    assert (gv > 0).all()
    return gv, names, labels


def yeo7_codes216():
    """Yeo-7 code 1..7 for every region, subcortex folded into its network."""
    labels = labels216()
    assign = subcortical_networks()
    return np.array([next(k for k, key in enumerate(YEO_KEYS) if key in lbl) + 1
                     if i < 200 else YEO_NAMES.index(assign[lbl]) + 1
                     for i, lbl in enumerate(labels)])
