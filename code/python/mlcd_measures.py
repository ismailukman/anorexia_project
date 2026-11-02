#!/usr/bin/env python3
"""
mlcd_measures.py: shared outcome-measure helpers for the dynamic-FC (MLCD) branch.

Holds the outcome-measure definitions used across the pipeline, so the
promiscuity denominator and the treatment of missing community labels are each
defined in exactly one place.

Missing community labels
------------------------
The multi-layer community detection output contains NaN where a node had no
valid community assignment in a layer, about 0.3% of cells per subject in this
data set. teneto treats NaN as an ordinary label, which distorts three of the
measures:

  * promiscuity  np.unique counts NaN as one more visited community, so almost
                 every node is credited with one community it never entered.
  * flexibility  NaN != anything is True, so a single missing layer is counted
                 as two community changes, one in and one out.
  * allegiance   NaN == NaN is False, so a missing layer counts as evidence
                 that two nodes were in different communities, when in fact
                 there is no evidence either way.

The functions here instead omit missing layers from both the numerator and the
denominator, so each measure is computed over the layers where it is defined.
Pass nan_aware=False to reproduce teneto's behaviour for comparison.

Why this exists
---------------
teneto.communitymeasures.promiscuity normalises by

    len(np.unique(C)) - 1

computed on whatever array it is handed. When C is one subject's slice of a
group MLCD run, that is the number of communities present in *that subject's*
layers, which here varies from 9 to 11 across subjects and differs
systematically between the anorexia and control runs. Two nodes that visit the
same number of communities then score differently for reasons unrelated to
their dynamics, so the measure is not comparable between subjects or groups.

promiscuity_runN() instead divides by a denominator fixed for the whole run,
which makes the values comparable across subjects. The formula is otherwise
identical to teneto's, so a node that never leaves one community still scores 0
and a node that visits every community of the run scores 1.
"""
import numpy as np

__all__ = ["allegiance", "recruitment", "integration", "flexibility",
           "match_community_affiliation",
           "run_community_count", "promiscuity_runN", "promiscuity_teneto"]


def allegiance(C, nan_aware=True):
    """Module allegiance matrix.

    P[i, j] is the fraction of layers in which nodes i and j carry the same
    community label; the diagonal is NaN.

    Parameters
    ----------
    C : array, (n_regions, n_layers)
        Community labels for one subject. NaN marks a layer in which a node
        had no community assignment.
    nan_aware : bool
        When True, each pair is scored over the layers in which both nodes
        have a label, so missing layers change neither the numerator nor the
        denominator. When False, every layer counts towards the denominator
        and a missing label counts as a mismatch, which reproduces
        teneto.communitymeasures.allegiance exactly.

    Returns
    -------
    array, (n_regions, n_regions)
        NaN for a pair that shares no jointly valid layer.

    Notes
    -----
    With nan_aware=False this is numerically identical to teneto's version but
    vectorised over node pairs, which matters because the reference
    implementation is a triple Python loop and is roughly 250 times slower at
    200-216 nodes and 663 layers.
    """
    C = np.asarray(C, dtype=float)
    n_nodes, n_layers = C.shape
    same = np.zeros((n_nodes, n_nodes), dtype=float)

    if not nan_aware:
        for t in range(n_layers):
            col = C[:, t]
            same += (col[:, None] == col[None, :])
        P = same / n_layers
        np.fill_diagonal(P, np.nan)
        return P

    valid = ~np.isnan(C)
    denom = np.zeros((n_nodes, n_nodes), dtype=float)
    for t in range(n_layers):
        col = C[:, t]
        ok = valid[:, t]
        both = ok[:, None] & ok[None, :]
        denom += both
        same += both & (col[:, None] == col[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(denom > 0, same / denom, np.nan)
    np.fill_diagonal(P, np.nan)
    return P


def run_community_count(group_matrices, nan_aware=True):
    """Number of distinct community labels in a whole MLCD run.

    With nan_aware=True, NaN is not counted as a community. Counting it adds
    one to the denominator of every promiscuity value.

    Parameters
    ----------
    group_matrices : sequence of arrays
        The full (n_regions, n_layers) community-label matrix of every group
        in the run, before any subject splitting or QC exclusion. The count is
        a property of the run's consensus partition, so it must not depend on
        which subjects a given analysis happens to keep.

    Returns
    -------
    int
    """
    labels = np.concatenate([np.unique(np.asarray(m, dtype=float))
                             for m in group_matrices])
    if nan_aware:
        labels = labels[~np.isnan(labels)]
    return int(len(np.unique(labels)))


def promiscuity_runN(C, run_ncomm, nan_aware=True):
    """Nodal promiscuity normalised by the run-wide community count.

    Parameters
    ----------
    C : array, (n_regions, n_layers)
        Community labels for one subject.
    run_ncomm : int
        Community count of the whole run, from run_community_count().
    nan_aware : bool
        When True, missing labels are not counted as a visited community.

    Returns
    -------
    array, (n_regions,)
        0 when a node stays in a single community, 1 when it visits all
        communities of the run. NaN for a node with no valid layer.
    """
    C = np.asarray(C, dtype=float)
    if run_ncomm < 2:
        raise ValueError(f"run_ncomm must be >= 2, got {run_ncomm}")
    out = np.empty(C.shape[0])
    for n in range(C.shape[0]):
        row = C[n]
        if nan_aware:
            row = row[~np.isnan(row)]
            if row.size == 0:
                out[n] = np.nan
                continue
        out[n] = len(np.unique(row)) - 1
    return out / (run_ncomm - 1)


def promiscuity_teneto(C):
    """teneto's original promiscuity: per-subject denominator, NaN as a label.

    Kept so the superseded values stay reproducible for comparison. Do not use
    it for between-subject or between-group tests; see the module docstring.
    """
    C = np.asarray(C)
    ncoms = len(np.unique(C)) - 1
    n_visited = np.array([len(np.unique(C[n])) for n in range(C.shape[0])])
    return (n_visited - 1) / ncoms


def flexibility(C, nan_aware=True):
    """Nodal flexibility: how often a node changes community between layers.

    Parameters
    ----------
    C : array, (n_regions, n_layers)
        Community labels for one subject.
    nan_aware : bool
        When True, each node's labelled layers are compared in sequence, so a
        missing layer closes the gap rather than creating one. A node observed
        as [1, 1, NaN, 2, 2] is read as [1, 1, 2, 2]: one change out of three
        transitions. Dropping the gap entirely would instead score it zero
        changes and lose a change that did happen, and teneto's handling
        (nan_aware=False) scores it two changes, one into the gap and one out.

    Returns
    -------
    array, (n_regions,)
        NaN for a node with fewer than two labelled layers.
    """
    C = np.asarray(C, dtype=float)
    if not nan_aware:
        return (C[:, 1:] != C[:, :-1]).sum(1) / (C.shape[1] - 1)
    out = np.empty(C.shape[0])
    for n in range(C.shape[0]):
        row = C[n]
        row = row[~np.isnan(row)]
        if row.size < 2:
            out[n] = np.nan
            continue
        out[n] = (row[1:] != row[:-1]).sum() / (row.size - 1)
    return out


def recruitment(C=None, static_communities=None, alleg=None):
    """Nodal recruitment: mean allegiance to nodes of the same static community.

    Parameters
    ----------
    C : array, (n_regions, n_layers), optional
        Community labels. Ignored when `alleg` is given.
    static_communities : array, (n_regions,)
        The a priori community of every node, e.g. its Yeo-7 network.
    alleg : array, (n_regions, n_regions), optional
        A precomputed allegiance matrix, to avoid recomputing it.

    Returns
    -------
    array, (n_regions,)

    Notes
    -----
    The node's own diagonal entry is NaN and is therefore excluded by the
    nanmean, matching teneto.communitymeasures.recruitment.
    """
    A = allegiance(C) if alleg is None else alleg
    sc = np.squeeze(np.asarray(static_communities))
    with np.errstate(invalid="ignore"):
        return np.array([np.nanmean(A[i, sc == s]) for i, s in enumerate(sc)])


def integration(C=None, static_communities=None, alleg=None):
    """Nodal integration: mean allegiance to nodes of every other static community.

    Parameters are as for recruitment(). Matches
    teneto.communitymeasures.integration.
    """
    A = allegiance(C) if alleg is None else alleg
    sc = np.squeeze(np.asarray(static_communities))
    with np.errstate(invalid="ignore"):
        return np.array([np.nanmean(A[i, sc != s]) for i, s in enumerate(sc)])


def match_community_affiliation(Ci):
    """Align community labels across adjacent layers.

    Community detection labels each layer independently, so the integer that
    names a community in one layer says nothing about the community with the
    same integer in the next. Flexibility and promiscuity read those integers
    directly and are meaningless until the labels are aligned; allegiance,
    recruitment and integration only ask whether two nodes share a label
    inside one layer and are unaffected either way.

    Each layer is matched to the already-matched layer before it: communities
    are paired greedily by how many nodes they share, largest overlap first,
    and a community with no overlap left to claim is given a new label.

    This is a port of match_community_affiliation.m from cocoanCORE
    (Jae-Joong Lee, 2021), which the MATLAB pipeline calls to produce
    `multi_module_consensus_sorted`. The saved `N_all_g` matrices hold the
    unaligned `multi_module_consensus` instead, so the alignment has to be
    applied here.

    Parameters
    ----------
    Ci : array, (n_nodes, n_layers)
        Community labels, NaN where a node has no assignment.

    Returns
    -------
    array, (n_nodes, n_layers)
        Labels aligned across layers. NaN is left in place.
    """
    Ci = np.asarray(Ci, dtype=float)
    out = Ci.copy()
    n_layers = Ci.shape[1]

    for t in range(1, n_layers):
        before = out[:, t - 1]
        after = Ci[:, t]
        u_before = np.unique(before[~np.isnan(before)])
        u_after = np.unique(after[~np.isnan(after)])
        if u_before.size == 0 or u_after.size == 0:
            continue

        # overlap[j, i] = nodes holding u_before[j] at t-1 and u_after[i] at t
        overlap = np.array([[np.sum((before == b) & (after == a))
                             for a in u_after] for b in u_before], dtype=float)

        # greedy one-to-one assignment, largest overlap first. The ordering
        # follows MATLAB's sort(overlap(:), 'descend'): column-major flatten,
        # stable so that equal overlaps keep their original order.
        mapped = np.full(u_after.size, np.nan)
        flat = overlap.ravel(order="F")
        order = np.argsort(-flat, kind="stable")
        rows, cols = np.unravel_index(order, overlap.shape, order="F")
        live = flat[order] > 0
        row_used = np.zeros(u_before.size, dtype=bool)
        col_used = np.zeros(u_after.size, dtype=bool)
        for r, c, ok in zip(rows, cols, live):
            if not ok or row_used[r] or col_used[c]:
                continue
            mapped[c] = u_before[r]
            row_used[r] = True
            col_used[c] = True

        # communities that matched nothing get fresh labels
        unmatched = np.isnan(mapped)
        if unmatched.any():
            mapped[unmatched] = u_before.max() + np.arange(1, unmatched.sum() + 1)

        col = out[:, t]
        for a, new in zip(u_after, mapped):
            col[after == a] = new
        out[:, t] = col

    return out
