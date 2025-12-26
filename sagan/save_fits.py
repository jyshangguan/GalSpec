from __future__ import annotations

import operator
import importlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import astropy
import astropy.units as u
from astropy.io import fits
from astropy.modeling.core import Model

import dill as pickle

__all__ = ["save_model", "load_model", "save_mcmc", "load_mcmc"]

# ----------------------------
# Simple pickle-based save/load
# ----------------------------

def save_model(model, filename):
    '''
    Save a model object to a file using dill for serialization.
    '''
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    '''
    Load a model object from a file using dill for serialization.
    '''
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def save_mcmc(mcmc, filename):
    '''
    Save an MCMC object to a file using dill for serialization.
    '''
    with open(filename, 'wb') as f:
        pickle.dump(mcmc, f)

def load_mcmc(filename):
    '''
    Load an MCMC object from a file using dill for serialization.
    '''
    with open(filename, 'rb') as f:
        mcmc = pickle.load(f)
    return mcmc

# ----------------------------
# Compound model utilities
# ----------------------------

# Accept both callable ops and string ops
_OP_TO_STR = {
    operator.add: "+",
    operator.mul: "*",
    operator.sub: "-",
    operator.truediv: "/",
    operator.pow: "**",
    "+": "+",
    "*": "*",
    "-": "-",
    "/": "/",
    "**": "**",
}

# Inverse mapping: string -> callable (for reconstruction)
_STR_TO_OP = {
    "+": operator.add,
    "*": operator.mul,
    "-": operator.sub,
    "/": operator.truediv,
    "**": operator.pow,
}

def _normalize_op_to_str(op):
    """Return canonical operator string for compound models."""
    # already a string like '+'
    if isinstance(op, str):
        if op in _STR_TO_OP:
            return op
        raise ValueError(f"Unsupported compound operator string {op!r}.")
    # callable case
    s = _OP_TO_STR.get(op)
    if s is None:
        raise ValueError(f"Unsupported compound operator {op!r}.")
    return s

def _is_compound(m: Model) -> bool:
    # Astropy CompoundModel has .left/.right/.op
    return hasattr(m, "left") and hasattr(m, "right") and hasattr(m, "op")


# ----------------------------
# Flatten model into FITS-friendly rows
# ----------------------------

def _flatten_model_tree(model: Model) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      nodes: list of dict rows for NODES table
      params: list of dict rows for PARAMS table

    Node fields:
      node_id, parent_id, role, kind, op, class_path, name

    Param fields:
      node_id, pname, value, unit, fixed, bound_lo, bound_hi
    """
    nodes: List[Dict[str, Any]] = []
    params: List[Dict[str, Any]] = []

    next_id = 0

    def new_id() -> int:
        nonlocal next_id
        nid = next_id
        next_id += 1
        return nid

    def visit(m: Model, parent_id: int, role: str) -> int:
        nid = new_id()

        if _is_compound(m):
            op_str = _normalize_op_to_str(m.op)

            nodes.append(
                dict(
                    node_id=nid,
                    parent_id=parent_id,
                    role=role,
                    kind="compound",
                    op=op_str,
                    class_path="",
                    name=getattr(m, "name", "") or "",
                )
            )
            # recurse
            visit(m.left, nid, "L")
            visit(m.right, nid, "R")
            return nid

        # leaf model
        cls = m.__class__
        nodes.append(
            dict(
                node_id=nid,
                parent_id=parent_id,
                role=role,
                kind="leaf",
                op="",
                class_path=f"{cls.__module__}.{cls.__name__}",
                name=getattr(m, "name", "") or "",
            )
        )

        # parameters
        for pname in m.param_names:
            p = getattr(m, pname)

            # store a single numeric VALUE plus UNIT string (if any)
            if p.unit is not None:
                val = float(p.quantity.to_value(p.unit))
                unit_str = str(p.unit)
            else:
                val = float(p.value)
                unit_str = ""

            fixed = bool(p.fixed)

            # bounds are (low, high); may contain None
            lo, hi = p.bounds
            lo = np.nan if lo is None else float(lo)
            hi = np.nan if hi is None else float(hi)

            params.append(
                dict(
                    node_id=nid,
                    pname=pname,
                    value=val,
                    unit=unit_str,
                    fixed=fixed,
                    bound_lo=lo,
                    bound_hi=hi,
                )
            )

        return nid

    # root has parent_id = -1, role = "ROOT"
    visit(model, parent_id=-1, role="ROOT")
    return nodes, params


# ----------------------------
# Rebuild model from rows
# ----------------------------

def _build_model_from_tables(nodes_tbl: fits.FITS_rec, params_tbl: fits.FITS_rec) -> Model:
    # Build node dict
    nodes = {}
    children = {}  # parent -> {"L": child, "R": child}
    for row in nodes_tbl:
        nid = int(row["NODE_ID"])
        pid = int(row["PARENT_ID"])
        role = row["ROLE"].strip()
        kind = row["KIND"].strip()
        op = row["OP"].strip()
        class_path = row["CLASS"].strip()
        name = row["NAME"].strip()

        nodes[nid] = dict(kind=kind, op=op, class_path=class_path, name=name, pid=pid, role=role)
        if pid >= 0:
            children.setdefault(pid, {})[role] = nid

    # Group params by node_id
    params_by_node: Dict[int, List[Dict[str, Any]]] = {}
    for row in params_tbl:
        nid = int(row["NODE_ID"])
        params_by_node.setdefault(nid, []).append(
            dict(
                pname=row["PNAME"].strip(),
                value=float(row["VALUE"]),
                unit=row["UNIT"].strip(),
                fixed=bool(row["FIXED"]),
                bound_lo=float(row["BLO"]),
                bound_hi=float(row["BHI"]),
            )
        )

    # Construct leaf instances first, then combine
    built: Dict[int, Model] = {}

    def build(nid: int) -> Model:
        if nid in built:
            return built[nid]

        info = nodes[nid]
        kind = info["kind"]

        if kind == "leaf":
            module_name, class_name = info["class_path"].rsplit(".", 1)
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)

            # Create a bare instance (often works)
            try:
                m = cls()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to construct model class {info['class_path']} with default constructor. "
                    f"Consider adding a custom factory for this class."
                ) from e

            # Apply parameters
            for pinfo in params_by_node.get(nid, []):
                pname = pinfo["pname"]
                p = getattr(m, pname)

                unit_str = pinfo["unit"]
                val = pinfo["value"]

                if unit_str:
                    q = val * u.Unit(unit_str)
                    # If parameter is unit-aware, set quantity
                    if getattr(p, "unit", None) is not None:
                        p.quantity = q
                    else:
                        # model param is unitless but we saved unit; drop unit
                        p.value = val
                else:
                    p.value = val

                p.fixed = bool(pinfo["fixed"])

                lo = pinfo["bound_lo"]
                hi = pinfo["bound_hi"]
                lo = None if np.isnan(lo) else lo
                hi = None if np.isnan(hi) else hi
                p.bounds = (lo, hi)

            if info["name"]:
                m.name = info["name"]

            built[nid] = m
            return m

        if kind == "compound":
            # Build children first
            kids = children.get(nid, {})
            if "L" not in kids or "R" not in kids:
                raise RuntimeError(f"Compound node {nid} is missing L/R children in table.")

            left = build(kids["L"])
            right = build(kids["R"])

            op = _STR_TO_OP.get(info["op"])
            if op is None:
                raise RuntimeError(f"Unsupported operator string {info['op']!r} in FITS table.")

            m = op(left, right)
            if info["name"]:
                m.name = info["name"]

            built[nid] = m
            return m

        raise RuntimeError(f"Unknown node kind {kind!r} in FITS table.")

    # Root is the node whose PARENT_ID == -1
    root_candidates = [int(r["NODE_ID"]) for r in nodes_tbl if int(r["PARENT_ID"]) == -1]
    if len(root_candidates) != 1:
        raise RuntimeError(f"Expected exactly one root node, found {root_candidates}.")
    return build(root_candidates[0])


# ----------------------------
# Public API
# ----------------------------

def save_model_fits(model: Model, filename: str, overwrite: bool = True) -> None:
    nodes, params = _flatten_model_tree(model)

    # Build NODES table
    node_id = np.array([r["node_id"] for r in nodes], dtype=np.int32)
    parent_id = np.array([r["parent_id"] for r in nodes], dtype=np.int32)
    role = np.array([r["role"] for r in nodes], dtype="S8")
    kind = np.array([r["kind"] for r in nodes], dtype="S8")
    op = np.array([r["op"] for r in nodes], dtype="S4")
    cls = np.array([r["class_path"] for r in nodes], dtype="S160")
    name = np.array([r["name"] for r in nodes], dtype="S80")

    hdu_nodes = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="NODE_ID", format="J", array=node_id),
            fits.Column(name="PARENT_ID", format="J", array=parent_id),
            fits.Column(name="ROLE", format="8A", array=role),
            fits.Column(name="KIND", format="8A", array=kind),
            fits.Column(name="OP", format="4A", array=op),
            fits.Column(name="CLASS", format="160A", array=cls),
            fits.Column(name="NAME", format="80A", array=name),
        ],
        name="NODES",
    )

    # Build PARAMS table
    if len(params) == 0:
        # still create an empty table
        hdu_params = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="NODE_ID", format="J", array=np.array([], dtype=np.int32)),
                fits.Column(name="PNAME", format="40A", array=np.array([], dtype="S40")),
                fits.Column(name="VALUE", format="D", array=np.array([], dtype=np.float64)),
                fits.Column(name="UNIT", format="40A", array=np.array([], dtype="S40")),
                fits.Column(name="FIXED", format="L", array=np.array([], dtype=np.bool_)),
                fits.Column(name="BLO", format="D", array=np.array([], dtype=np.float64)),
                fits.Column(name="BHI", format="D", array=np.array([], dtype=np.float64)),
            ],
            name="PARAMS",
        )
    else:
        hdu_params = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="NODE_ID", format="J", array=np.array([r["node_id"] for r in params], dtype=np.int32)),
                fits.Column(name="PNAME", format="40A", array=np.array([r["pname"] for r in params], dtype="S40")),
                fits.Column(name="VALUE", format="D", array=np.array([r["value"] for r in params], dtype=np.float64)),
                fits.Column(name="UNIT", format="40A", array=np.array([r["unit"] for r in params], dtype="S40")),
                fits.Column(name="FIXED", format="L", array=np.array([r["fixed"] for r in params], dtype=np.bool_)),
                fits.Column(name="BLO", format="D", array=np.array([r["bound_lo"] for r in params], dtype=np.float64)),
                fits.Column(name="BHI", format="D", array=np.array([r["bound_hi"] for r in params], dtype=np.float64)),
            ],
            name="PARAMS",
        )

    # Primary header metadata
    hdr = fits.Header()
    hdr["ORIGIN"] = "astropy.modeling"
    hdr["ASTROPY"] = astropy.__version__
    hdr["FORMAT"] = "AMODFITS"   # "Astropy MODel FITS"
    hdr["FVER"] = 1
    phdu = fits.PrimaryHDU(header=hdr)

    fits.HDUList([phdu, hdu_nodes, hdu_params]).writeto(filename, overwrite=overwrite)


def load_model_fits(filename: str) -> Model:
    with fits.open(filename) as hdul:
        nodes_tbl = hdul["NODES"].data
        params_tbl = hdul["PARAMS"].data
        return _build_model_from_tables(nodes_tbl, params_tbl)
