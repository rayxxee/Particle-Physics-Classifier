"""
uproot-based ROOT file reader.

Reads CMS/ATLAS ROOT files without requiring a native ROOT installation.
Extracts TTree branches into pandas DataFrames, handling variable-length
arrays (e.g., variable number of jets per event) via padding/masking.

Usage:
    from src.ingestion.root_reader import RootReader

    reader = RootReader("data/raw/cms_run2.root", tree_name="Events")
    df = reader.to_dataframe(max_events=100_000)

Note:
    ROOT data is used in Phase 2+. In Phase 1, the HIGGS CSV reader is primary.
    This module is fully implemented but not required for the initial MLP training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

log = get_logger(__name__)

# ─── Branch configuration ────────────────────────────────────────────────────

# Default branch names for CMS NanoAOD format (Run 2 open data)
DEFAULT_BRANCHES = {
    # Muons
    "muon_pt": "Muon_pt",
    "muon_eta": "Muon_eta",
    "muon_phi": "Muon_phi",
    "muon_mass": "Muon_mass",
    "n_muons": "nMuon",
    # Electrons
    "electron_pt": "Electron_pt",
    "electron_eta": "Electron_eta",
    "electron_phi": "Electron_phi",
    "electron_mass": "Electron_mass",
    "n_electrons": "nElectron",
    # Jets (AK4)
    "jet_pt": "Jet_pt",
    "jet_eta": "Jet_eta",
    "jet_phi": "Jet_phi",
    "jet_mass": "Jet_mass",
    "jet_btag": "Jet_btagDeepFlavB",
    "n_jets": "nJet",
    # MET
    "met_pt": "MET_pt",
    "met_phi": "MET_phi",
}

MAX_JETS = 4       # Pad/truncate jet arrays to this length
MAX_LEPTONS = 2    # Pad/truncate lepton arrays to this length


class RootReader:
    """
    Reads a single ROOT file and converts TTree branches to a pandas DataFrame.

    Args:
        filepath:    Path to the .root file.
        tree_name:   Name of the TTree to read (default: "Events" for NanoAOD).
        branches:    Mapping of local names → ROOT branch names.
                     If None, uses DEFAULT_BRANCHES.

    Example:
        reader = RootReader("data/raw/cms_run2.root")
        df = reader.to_dataframe(max_events=50_000)
    """

    def __init__(
        self,
        filepath: str | Path,
        tree_name: str = "Events",
        branches: dict[str, str] | None = None,
    ) -> None:
        self.filepath = Path(filepath)
        self.tree_name = tree_name
        self.branches = branches or DEFAULT_BRANCHES

        if not self.filepath.exists():
            raise FileNotFoundError(f"ROOT file not found: {self.filepath}")

        log.info(
            "RootReader initialized",
            filepath=str(self.filepath),
            tree=tree_name,
            file_size_mb=round(self.filepath.stat().st_size / 1e6, 1),
        )

    def _open_tree(self) -> Any:
        """Open the TTree. Imports uproot lazily to avoid hard dependency."""
        try:
            import uproot
        except ImportError as e:
            raise ImportError(
                "uproot is required to read ROOT files. "
                "Install with: pip install uproot awkward"
            ) from e

        root_file = uproot.open(str(self.filepath))

        # Handle both "TreeName" and "TreeName;1" (TKey versioning)
        available = list(root_file.keys())
        if self.tree_name not in available:
            candidates = [k for k in available if k.startswith(self.tree_name)]
            if not candidates:
                raise ValueError(
                    f"Tree '{self.tree_name}' not found in {self.filepath}. "
                    f"Available: {available}"
                )
            self.tree_name = candidates[0]
            log.debug("Resolved tree name", name=self.tree_name)

        return root_file[self.tree_name]

    def _pad_array(
        self, array: Any, max_len: int, fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Pad or truncate a variable-length awkward array to a fixed length.

        For events with fewer than max_len particles, fills with fill_value.
        For events with more than max_len, takes the first max_len (leading).
        """
        import awkward as ak

        # Pad to max_len, clip from right
        padded = ak.pad_none(array, max_len, clip=True)
        # Fill None with fill_value
        filled = ak.fill_none(padded, fill_value)
        return ak.to_numpy(filled)

    def to_dataframe(
        self,
        max_events: int | None = None,
        apply_quality_cuts: bool = True,
    ) -> pd.DataFrame:
        """
        Read the ROOT file and return a flat pandas DataFrame.

        Variable-length arrays (jets, leptons) are padded to fixed width.
        Column names follow the HIGGS dataset convention for compatibility.

        Args:
            max_events:          Maximum number of events to read. None = all.
            apply_quality_cuts:  Apply standard HEP selection cuts.

        Returns:
            DataFrame with columns matching the HIGGS schema.
        """
        tree = self._open_tree()
        log.info(
            "Reading TTree",
            n_entries=tree.num_entries,
            max_events=max_events,
        )

        entry_stop = max_events  # uproot uses entry_stop for slicing

        # Read all branches as awkward arrays
        arrays = tree.arrays(
            list(self.branches.values()),
            entry_stop=entry_stop,
            library="ak",
        )

        rows: dict[str, np.ndarray] = {}

        # ── Scalar branches ───────────────────────────────────────────────────
        import awkward as ak

        def _to_numpy(branch_name: str) -> np.ndarray:
            local_name = _find_local(branch_name)
            return ak.to_numpy(arrays[self.branches[local_name]])

        for local, root_branch in self.branches.items():
            arr = arrays[root_branch]
            # Variable-length arrays handled separately
            if arr.ndim == 1:
                rows[local] = ak.to_numpy(arr).astype(np.float32)

        # ── Variable-length: jets ─────────────────────────────────────────────
        for jet_feat in ["jet_pt", "jet_eta", "jet_phi", "jet_mass", "jet_btag"]:
            if jet_feat in self.branches:
                padded = self._pad_array(arrays[self.branches[jet_feat]], MAX_JETS)
                for j in range(MAX_JETS):
                    col_name = jet_feat.replace("jet_", f"jet{j+1}_")
                    rows[col_name] = padded[:, j].astype(np.float32)

        # ── Variable-length: leptons (take leading) ───────────────────────────
        for lep_feat in ["muon_pt", "muon_eta", "muon_phi", "electron_pt", "electron_eta", "electron_phi"]:
            if lep_feat in self.branches:
                padded = self._pad_array(arrays[self.branches[lep_feat]], MAX_LEPTONS)
                rows[lep_feat.replace("_", "_leading_", 1) + "_0"] = padded[:, 0].astype(np.float32)

        df = pd.DataFrame(rows)

        if apply_quality_cuts:
            df = self._apply_quality_cuts(df)

        log.info(
            "RootReader complete",
            n_events=len(df),
            n_cols=len(df.columns),
        )
        return df

    def _apply_quality_cuts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply standard HEP event selection cuts."""
        original_len = len(df)

        mask = pd.Series(True, index=df.index)

        # Lepton pT cut
        if "muon_pt" in df.columns:
            mask &= df["muon_pt"] > 25.0
        if "electron_pt" in df.columns:
            mask &= df["electron_pt"] > 25.0

        # MET cut
        if "met_pt" in df.columns:
            mask &= df["met_pt"] > 20.0

        # Jet pT cut (at least 2 jets above threshold)
        jet_pt_cols = [c for c in df.columns if c.startswith("jet") and c.endswith("_pt")]
        if jet_pt_cols:
            n_jets_passing = sum(df[c] > 30.0 for c in jet_pt_cols)
            mask &= n_jets_passing >= 2

        df = df[mask].reset_index(drop=True)
        log.info(
            "Quality cuts applied",
            before=original_len,
            after=len(df),
            efficiency=f"{len(df)/original_len:.1%}",
        )
        return df

    def list_branches(self) -> list[str]:
        """Return all branch names in the TTree."""
        tree = self._open_tree()
        return list(tree.keys())

    def inspect(self) -> None:
        """Print a summary of the ROOT file content."""
        tree = self._open_tree()
        print(f"\nROOT file: {self.filepath}")
        print(f"Tree: {self.tree_name} ({tree.num_entries:,} entries)")
        print(f"Branches ({len(list(tree.keys()))}):")
        for branch in list(tree.keys())[:30]:
            print(f"  {branch}")
        if tree.num_entries > 30:
            print("  ...")


def _find_local(branch_name: str) -> str:
    """Reverse lookup: ROOT branch name → local name key."""
    return branch_name  # placeholder — used by to_dataframe internally
