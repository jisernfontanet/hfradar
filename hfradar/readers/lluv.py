"""LLUV file reader (radials, ellipticals, totals)."""

from __future__ import annotations

import gzip
import io

import numpy as np

from . ctf import _parse_ctf_stream

# Columns affected by %XYUnits: scaling
_XY_COLS = {"xdst", "ydst", "rnge"}
# Columns affected by %UVUnits: scaling
_UV_COLS = {"velu", "velv", "velo", "maxv", "minv"}

# Known LLUV subtype prefixes (from primary table type string)
_SUBTYPE_MAP = {
    "rdl": "rdls",
    "elp": "elps",
    "tot": "tots",
}


def _detect_gzip(filename: str) -> bool:
    with open(filename, "rb") as f:
        magic = f.read(2)
    return magic == b"\x1f\x8b"


def _open_text(filename: str) -> io.StringIO:
    if _detect_gzip(filename):
        with gzip.open(filename, "rb") as f:
            raw = f.read()
    else:
        with open(filename, "rb") as f:
            raw = f.read()
    return io.StringIO(raw.decode("latin-1"))


def _lluv_subtype(filetype_value: str, primary_table_type: str) -> str:
    """Determine the LLUV subtype string."""
    tokens = filetype_value.split()
    if len(tokens) >= 2:
        return tokens[1].lower()
    table_tokens = primary_table_type.split()
    if len(table_tokens) >= 2:
        prefix = table_tokens[1][:3].lower()
        return _SUBTYPE_MAP.get(prefix, prefix)
    return ""


def read_lluv(filename: str) -> dict:
    """Read a CODAR LLUV (Lon/Lat/U/V) file.

    Handles radials (``.ruv``), ellipticals (``.euv``), and totals (``.tuv``).
    Gzip-compressed variants are detected by magic bytes (``0x1f 0x8b``) and
    decompressed transparently, regardless of file extension.

    The primary LLUV table (the first table whose ``%TableType:`` begins with
    ``LLUV``) is split into per-column 1-D arrays stored directly in
    ``data``, keyed by the lowercase four-character column code from
    ``%TableColumnTypes:``. If ``%XYUnits:`` or ``%UVUnits:`` appear before
    the table, the corresponding columns are scaled to metres and m/s
    respectively. All remaining tables are collected in
    ``data["secondary_tables"]``.

    Args:
        filename: Path to the LLUV file (plain text or gzip-compressed).
            All metadata is read from file contents; no information is
            derived from the filename string.

    Returns:
        A dict with two keys:

        - ``"metadata"`` (dict): all CTF keyword metadata (original-case
          keys) plus the synthetic key ``"lluv_subtype"`` (``str``), which
          is one of ``"rdls"``, ``"elps"``, or ``"tots"``.
        - ``"data"`` (dict): one ``numpy.ndarray`` of shape ``(nVectors,)``
          and dtype ``float64`` per column in the primary LLUV table, keyed
          by its lowercase four-character code (e.g. ``"lond"``, ``"latd"``,
          ``"velu"``, ``"velv"``), plus:

          - ``"secondary_tables"`` (numpy.ndarray, dtype ``object``):
            1-D object array of table dicts for non-primary tables (e.g.
            diagnostic or source tables). Each dict has ``"table_type"``,
            ``"column_types"``, and ``"data"`` (ndarray, float64).

    Raises:
        FileNotFoundError: If ``filename`` does not exist.
        ValueError: If ``%FileType:`` is missing or its type token is not
            ``LLUV`` (case-insensitive).

    Example:
        >>> result = read_lluv("RDLm_SITE_2024_01_01_1200.ruv")
        >>> result["metadata"]["lluv_subtype"]
        'rdls'
        >>> result["data"]["lond"].shape
        (627,)
        >>> result["data"]["velu"].dtype
        dtype('float64')
    """
    stream = _open_text(filename)
    parsed = _parse_ctf_stream(stream)

    meta = parsed["metadata"]
    tables = parsed["data"]["tables"]

    # Validate FileType
    filetype_value: str = meta.get("FileType", "")
    if not filetype_value.split()[0:1] or filetype_value.split()[0].upper() != "LLUV":
        raise ValueError(
            f"Expected %FileType: LLUV ..., got {filetype_value!r}. "
            "This does not appear to be an LLUV file."
        )

    # Identify primary LLUV table (first table whose type starts with "LLUV")
    primary_table = None
    secondary_tables = []
    for t in tables:
        tt = t.get("table_type", "").upper()
        if tt.startswith("LLUV") and primary_table is None:
            primary_table = t
        else:
            secondary_tables.append(t)

    data: dict[str, np.ndarray] = {}

    if primary_table is not None:
        col_types = primary_table.get("column_types", [])
        table_data: np.ndarray = primary_table["data"]

        # XY and UV unit scalars (parsed from metadata before the table)
        xy_scalar = 1.0
        uv_scalar = 1.0
        if "XYUnits" in meta:
            parts = meta["XYUnits"].split()
            if len(parts) >= 2:
                try:
                    xy_scalar = float(parts[1])
                except ValueError:
                    pass
        if "UVUnits" in meta:
            parts = meta["UVUnits"].split()
            if len(parts) >= 2:
                try:
                    uv_scalar = float(parts[1])
                except ValueError:
                    pass

        # Slice each column into a 1-D float64 array
        for col_idx, code in enumerate(col_types):
            key = code.lower()
            if col_idx < table_data.shape[1]:
                col = table_data[:, col_idx].astype(np.float64)
                if key in _XY_COLS:
                    col = col * xy_scalar
                elif key in _UV_COLS:
                    col = col * uv_scalar
                data[key] = col

    # Secondary tables as object array of dicts
    data["secondary_tables"] = np.array(secondary_tables, dtype=object)

    # Determine subtype and store in metadata
    primary_type = primary_table["table_type"] if primary_table else ""
    meta["lluv_subtype"] = _lluv_subtype(filetype_value, primary_type)

    return {"metadata": meta, "data": data}
