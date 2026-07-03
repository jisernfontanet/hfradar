"""
CODAR SeaSonde Cross Spectra file reader.

Supports:
  - Standard Cross Spectra files (Cross Spectra File Format Version 6)
  - Reduced Cross Spectra files (RIFF-based compressed format)

Public API
----------
    read_codar_cross_spectra(filename: str) -> dict
"""

import io
import math
import struct
from datetime import datetime, timezone

import numpy as np

# Seconds from CODAR/Mac epoch (1904-01-01 UTC) to Unix epoch (1970-01-01 UTC)
_CODAR_EPOCH = 2_082_844_800

# Magic bytes that identify a Reduced CSS file (big-endian and little-endian variants)
_REDUCED_MAGIC_BE = {b"CSSW", b"CSSY"}
_REDUCED_MAGIC_LE = {b"WSSC", b"YSSC"}
_REDUCED_MAGIC = _REDUCED_MAGIC_BE | _REDUCED_MAGIC_LE


# ─── Public API ───────────────────────────────────────────────────────────────

def read_cross_spectra(filename: str) -> dict:
    """Read a CODAR SeaSonde Cross Spectra file.

    Auto-detects standard vs. reduced (RIFF-compressed) format.

    Parameters
    ----------
    filename : str
        Path to a .cs or .csr file.

    Returns
    -------
    dict
        file_type : 'standard' or 'reduced'
        header    : dict of all header metadata fields
        v6_blocks : dict of decoded Version 6 metadata blocks (when present)
        data      : dict of NumPy arrays shaped (nRangeCells, nDopplerCells)
            antenna1_self  float32   voltage-squared self spectra, ant 1
            antenna2_self  float32   voltage-squared self spectra, ant 2
            antenna3_self  float32   voltage-squared self spectra, ant 3
                                     (may contain negatives per CODAR convention)
            cross_12       complex64 cross spectra ant 1 → 2
            cross_13       complex64 cross spectra ant 1 → 3
            cross_23       complex64 cross spectra ant 2 → 3
            quality        float32   averaging quality (present when nCsKind ≥ 2
                                     or in all reduced files)
    """
    with open(filename, "rb") as f:
        magic = f.read(4)
        f.seek(0)
        if magic in _REDUCED_MAGIC:
            return _read_reduced(f)
        return _read_standard(f)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _codar_ts(secs: int) -> datetime:
    return datetime.fromtimestamp(secs - _CODAR_EPOCH, tz=timezone.utc)


def _rb(f, fmt):
    """Unpack big-endian struct from file, return tuple."""
    s = ">" + fmt
    return struct.unpack(s, f.read(struct.calcsize(s)))


def _rb1(f, fmt):
    return _rb(f, fmt)[0]


def _drop_private(d: dict):
    for k in [k for k in d if k.startswith("_")]:
        del d[k]


# ─── Standard CSS reader ──────────────────────────────────────────────────────

def _read_standard(f) -> dict:
    version  = _rb1(f, "h")   # SInt16
    dt_raw   = _rb1(f, "I")   # UInt32
    v1extent = _rb1(f, "i")   # SInt32

    if not (1 <= version <= 32):
        raise ValueError(f"Not a valid CSS file (nCsFileVersion={version})")

    header = {
        "nCsFileVersion": version,
        "nDateTime":      dt_raw,
        "datetime":       _codar_ts(dt_raw),
        "nV1Extent":      v1extent,
    }

    # Defaults for version < 4
    n_cs_kind          = 1
    n_range_cells      = 31
    n_doppler_cells    = 512
    n_spectra_channels = 3

    if version >= 2:
        header["nCsKind"]   = _rb1(f, "h")
        header["nV2Extent"] = _rb1(f, "i")
        n_cs_kind = header["nCsKind"]

    if version >= 3:
        header["nSiteCodeName"] = f.read(4).rstrip(b"\x00").decode("ascii", errors="replace")
        header["nV3Extent"]     = _rb1(f, "i")

    if version >= 4:
        vals = _rb(f, "iiifffiiiifi")
        keys = (
            "nCoverMinutes", "bDeletedSource", "bOverrideSrcInfo",
            "fStartFreqMHz", "fRepFreqHz", "fBandwidthKHz", "bSweepUp",
            "nDopplerCells", "nRangeCells", "nFirstRangeCell",
            "fRangeCellDistKm", "nV4Extent",
        )
        header.update(zip(keys, vals))
        n_doppler_cells   = header["nDopplerCells"]
        n_range_cells     = header["nRangeCells"]

    if version >= 5:
        header["nOutputInterval"]  = _rb1(f, "i")
        header["nCreateTypeCode"]  = f.read(4).rstrip(b"\x00").decode("ascii", errors="replace")
        header["nCreatorVersion"]  = f.read(4).rstrip(b"\x00").decode("ascii", errors="replace")
        header["nActiveChannels"]  = _rb1(f, "i")
        header["nSpectraChannels"] = _rb1(f, "i")
        header["nActiveChanBits"]  = _rb1(f, "I")
        header["nV5Extent"]        = _rb1(f, "i")
        n_spectra_channels = header["nSpectraChannels"]

    v6_blocks = {}
    if version >= 6:
        cs6_size = _rb1(f, "I")
        header["nCS6ByteSize"] = cs6_size
        remaining = cs6_size
        while remaining > 0:
            key      = f.read(4).decode("ascii", errors="replace")
            blk_size = _rb1(f, "I")
            raw      = f.read(blk_size)
            parsed = _parse_v6_block(key, raw, n_range_cells, n_doppler_cells, n_spectra_channels)
            # TOOL may appear multiple times (one per tool that processed the data)
            if key == "TOOL":
                v6_blocks.setdefault("TOOL", []).append(parsed)
            else:
                v6_blocks[key] = parsed
            remaining -= 8 + blk_size

    # Jump to data section using nV1Extent
    f.seek(v1extent + 10)
    data = _read_standard_data(f, n_range_cells, n_doppler_cells, n_spectra_channels, n_cs_kind)

    return {
        "file_type": "standard",
        "header":    header,
        "v6_blocks": v6_blocks,
        "data":      data,
    }


def _parse_v6_block(key: str, raw: bytes, n_ranges: int, n_dopplers: int, n_channels: int) -> dict:
    n = len(raw)

    if key == "TIME" and n >= 31:
        mark, year, mo, day, hr, mn = struct.unpack_from(">BHBBBB", raw)
        fsec, fcov, futc = struct.unpack_from(">3d", raw, 7)
        return {
            "nTimeMark": mark,
            "nYear": year, "nMonth": mo, "nDay": day,
            "nHour": hr, "nMinute": mn,
            "fSeconds": fsec, "fCoverageSeconds": fcov, "fHoursFromUTC": futc,
        }

    if key in ("ZONE", "CITY", "SITD", "TOOL"):
        return {"value": raw.rstrip(b"\x00").decode("ascii", errors="replace")}

    if key == "LOCA" and n >= 24:
        lat, lon, alt = struct.unpack_from(">3d", raw)
        return {"fLatitude": lat, "fLongitude": lon, "fAltitudeMeters": alt}

    if key == "RCVI" and n >= 48:
        mdl, ant_mdl = struct.unpack_from(">II", raw)
        (gain,)      = struct.unpack_from(">d", raw, 8)
        fw = raw[16:48].rstrip(b"\x00").decode("ascii", errors="replace")
        return {
            "nReceiverModel": mdl, "nRxAntennaModel": ant_mdl,
            "fReferenceGainDB": gain, "szFirmware": fw,
        }

    if key == "GLRM" and n >= 39:
        mth, ver           = struct.unpack_from(">BB", raw)
        pts, tms, segs     = struct.unpack_from(">3I", raw, 2)
        pt_p, rng_p, rng_b = struct.unpack_from(">3d", raw, 14)
        dc = raw[38]
        return {
            "nMethod": mth, "nVersion": ver,
            "nPointsRemoved": pts, "nTimesRemoved": tms, "nSegmentsRemoved": segs,
            "fPointPowerThreshold": pt_p, "fRangePowerThreshold": rng_p,
            "fRangeBinThreshold": rng_b, "bRemoveDC": bool(dc),
        }

    if key == "SUPI" and n >= 28:
        mth, ver, mode, dbg = struct.unpack_from(">4B", raw)
        (n_sup,)            = struct.unpack_from(">I",  raw, 4)
        pwr, rng_b          = struct.unpack_from(">2d", raw, 8)
        r_band, d_smooth    = struct.unpack_from(">2h", raw, 24)
        return {
            "nMethod": mth, "nVersion": ver, "nMode": mode, "nDebugMode": dbg,
            "nDopplerSuppressed": n_sup,
            "fPowerThreshold": pwr, "fRangeBinThreshold": rng_b,
            "nRangeBanding": r_band, "nDopplerDetectionSmoothing": d_smooth,
        }

    if key == "SUPM":
        nb = n_channels * n_dopplers * 4
        if n >= nb:
            return {
                "fSuppressionVoltageSquared": np.frombuffer(
                    raw[:nb], dtype=">f4"
                ).reshape(n_channels, n_dopplers).copy()
            }

    if key == "SUPP":
        nb = n_channels * n_dopplers * 4
        if n >= nb:
            return {
                "fPhaseDegrees": np.frombuffer(
                    raw[:nb], dtype=">f4"
                ).reshape(n_channels, n_dopplers).copy()
            }

    if key == "ANTG":
        nb = n_channels * 8
        if n >= nb:
            return {"fGainDB": np.frombuffer(raw[:nb], dtype=">f8").copy()}

    if key == "FWIN" and n >= 18:
        rw, dw  = struct.unpack_from(">2B", raw)
        rp, dp  = struct.unpack_from(">2d", raw, 2)
        return {
            "nRangeWindowType": rw, "nDopplerWindowType": dw,
            "fRangeWindowParam": rp, "fDopplerWindowParam": dp,
        }

    if key == "IQAP" and n >= 2:
        mth, ver = struct.unpack_from(">2B", raw)
        n_pairs  = (n - 2) // 16
        pairs    = np.frombuffer(raw[2: 2 + n_pairs * 16], dtype=">f8").reshape(n_pairs, 2).copy()
        return {"nMethod": mth, "nVersion": ver, "data": pairs}

    if key == "FILL" and n >= 4:
        rm, rmt, dm, dmt = struct.unpack_from(">4B", raw)
        return {
            "nRangeMethod": rm, "nRangeMult": rmt,
            "nDopplerMethod": dm, "nDopplerMult": dmt,
        }

    if key in ("FOLS", "WOLS"):
        nb = n_ranges * 16
        if n >= nb:
            arr = np.frombuffer(raw[:nb], dtype=">i4").reshape(n_ranges, 4).copy()
            return {
                "nNegBraggLeftIndex":  arr[:, 0],
                "nNegBraggRightIndex": arr[:, 1],
                "nPosBraggLeftIndex":  arr[:, 2],
                "nPosBraggRightIndex": arr[:, 3],
            }

    if key == "BRGR" and n >= n_ranges:
        return {"nBraggReject": np.frombuffer(raw[:n_ranges], dtype="u1").copy()}

    return {"raw": raw}


def _read_standard_data(
    f, n_ranges: int, n_dopplers: int, n_channels: int, cs_kind: int
) -> dict:
    self1  = np.empty((n_ranges, n_dopplers), dtype=np.float32)
    self2  = np.empty((n_ranges, n_dopplers), dtype=np.float32)
    self3  = np.empty((n_ranges, n_dopplers), dtype=np.float32)
    c12    = np.empty((n_ranges, n_dopplers), dtype=np.complex64)
    c13    = np.empty((n_ranges, n_dopplers), dtype=np.complex64)
    c23    = np.empty((n_ranges, n_dopplers), dtype=np.complex64)
    qual   = np.empty((n_ranges, n_dopplers), dtype=np.float32) if cs_kind >= 2 else None

    bpr = n_dopplers * 4  # bytes per real row

    for r in range(n_ranges):
        self1[r] = np.frombuffer(f.read(bpr),          dtype=">f4")
        self2[r] = np.frombuffer(f.read(bpr),          dtype=">f4")
        self3[r] = np.frombuffer(f.read(bpr),          dtype=">f4")

        raw12 = np.frombuffer(f.read(n_dopplers * 8), dtype=">f4").reshape(n_dopplers, 2)
        c12[r] = raw12[:, 0].astype(np.float32) + 1j * raw12[:, 1].astype(np.float32)

        raw13 = np.frombuffer(f.read(n_dopplers * 8), dtype=">f4").reshape(n_dopplers, 2)
        c13[r] = raw13[:, 0].astype(np.float32) + 1j * raw13[:, 1].astype(np.float32)

        raw23 = np.frombuffer(f.read(n_dopplers * 8), dtype=">f4").reshape(n_dopplers, 2)
        c23[r] = raw23[:, 0].astype(np.float32) + 1j * raw23[:, 1].astype(np.float32)

        if qual is not None:
            qual[r] = np.frombuffer(f.read(bpr), dtype=">f4")

    out = {
        "antenna1_self": self1,
        "antenna2_self": self2,
        "antenna3_self": self3,
        "cross_12":      c12,
        "cross_13":      c13,
        "cross_23":      c23,
    }
    if qual is not None:
        out["quality"] = qual
    return out


# ─── Reduced CSS reader ───────────────────────────────────────────────────────

def _sint24_be(b: bytes, pos: int) -> int:
    v = (b[pos] << 16) | (b[pos + 1] << 8) | b[pos + 2]
    return v - 0x1000000 if v >= 0x800000 else v


def _sint24_le(b: bytes, pos: int) -> int:
    v = b[pos] | (b[pos + 1] << 8) | (b[pos + 2] << 16)
    return v - 0x1000000 if v >= 0x800000 else v


def _decode_cssw(encoded: bytes, n_dopplers: int, big_endian: bool) -> np.ndarray:
    """Decode a CSSW variable-length encoded byte stream to a UInt32 array."""
    endian  = ">" if big_endian else "<"
    sint24  = _sint24_be if big_endian else _sint24_le
    u32_fmt = endian + "I"
    s16_fmt = endian + "h"
    s8_fmt  = endian + "b"

    output   = []
    tracking = 0
    pos      = 0
    b        = encoded

    while pos < len(b) and len(output) < n_dopplers:
        cmd = b[pos]; pos += 1

        if cmd == 0x9C:                          # single UInt32 absolute
            tracking = struct.unpack_from(u32_fmt, b, pos)[0]; pos += 4
            output.append(tracking)

        elif cmd == 0x94:                        # run of UInt32 absolutes
            count = b[pos] + 1; pos += 1
            for _ in range(count):
                tracking = struct.unpack_from(u32_fmt, b, pos)[0]; pos += 4
                output.append(tracking)

        elif cmd == 0xAC:                        # single SInt24 delta
            delta = sint24(b, pos); pos += 3
            tracking = (tracking + delta) & 0xFFFFFFFF
            output.append(tracking)

        elif cmd == 0xA4:                        # run of SInt24 deltas
            count = b[pos] + 1; pos += 1
            for _ in range(count):
                delta = sint24(b, pos); pos += 3
                tracking = (tracking + delta) & 0xFFFFFFFF
                output.append(tracking)

        elif cmd == 0x89:                        # single SInt8 delta
            delta = struct.unpack_from(s8_fmt, b, pos)[0]; pos += 1
            tracking = (tracking + delta) & 0xFFFFFFFF
            output.append(tracking)

        elif cmd == 0x84:                        # single SInt16 delta
            delta = struct.unpack_from(s16_fmt, b, pos)[0]; pos += 2
            tracking = (tracking + delta) & 0xFFFFFFFF
            output.append(tracking)

        elif cmd == 0x82:                        # run of SInt16 deltas
            count = b[pos] + 1; pos += 1
            for _ in range(count):
                delta = struct.unpack_from(s16_fmt, b, pos)[0]; pos += 2
                tracking = (tracking + delta) & 0xFFFFFFFF
                output.append(tracking)

        elif cmd == 0x81:                        # run of SInt8 deltas
            count = b[pos] + 1; pos += 1
            for _ in range(count):
                delta = struct.unpack_from(s8_fmt, b, pos)[0]; pos += 1
                tracking = (tracking + delta) & 0xFFFFFFFF
                output.append(tracking)

        elif cmd == 0x8A:                        # single SInt16 delta
            delta = struct.unpack_from(s16_fmt, b, pos)[0]; pos += 2
            tracking = (tracking + delta) & 0xFFFFFFFF
            output.append(tracking)

        else:
            raise ValueError(f"Unknown CSSW command byte 0x{cmd:02X} at offset {pos - 1}")

    return np.array(output[:n_dopplers], dtype=np.uint32)


def _apply_scal(raw: np.ndarray, fmin: float, fmax: float, fscale: float) -> np.ndarray:
    """Map UInt32 fixed-point values to floats using scal parameters."""
    return np.where(
        raw == 0xFFFFFFFF,
        np.nan,
        raw.astype(np.float64) * (fmax - fmin) / fscale + fmin,
    )


def _db_to_power(db: np.ndarray, db_ref: float) -> np.ndarray:
    """Convert dB-relative values to power (V²) using reference gain."""
    return np.power(10.0, (db + db_ref) / 10.0)


def _parse_cs4h(data: bytes, endian: str) -> tuple:
    """Parse the cs4h embedded standard CSS header from a bytes buffer.

    Returns (header_dict, v6_blocks_dict).
    """
    f = io.BytesIO(data)

    def u1(fmt):
        s = endian + fmt
        return struct.unpack(s, f.read(struct.calcsize(s)))[0]

    def u(fmt):
        s = endian + fmt
        return struct.unpack(s, f.read(struct.calcsize(s)))

    version  = u1("h")
    dt_raw   = u1("I")
    v1extent = u1("i")

    header = {
        "nCsFileVersion": version,
        "nDateTime":      dt_raw,
        "datetime":       _codar_ts(dt_raw),
        "nV1Extent":      v1extent,
    }

    n_cs_kind          = 1
    n_range_cells      = 31
    n_doppler_cells    = 512
    n_spectra_channels = 3

    if version >= 2:
        header["nCsKind"]   = u1("h")
        header["nV2Extent"] = u1("i")
        n_cs_kind = header["nCsKind"]

    if version >= 3:
        header["nSiteCodeName"] = f.read(4).rstrip(b"\x00").decode("ascii", errors="replace")
        header["nV3Extent"]     = u1("i")

    if version >= 4:
        vals = u("iiifffiiiifi")
        keys = (
            "nCoverMinutes", "bDeletedSource", "bOverrideSrcInfo",
            "fStartFreqMHz", "fRepFreqHz", "fBandwidthKHz", "bSweepUp",
            "nDopplerCells", "nRangeCells", "nFirstRangeCell",
            "fRangeCellDistKm", "nV4Extent",
        )
        header.update(zip(keys, vals))
        n_doppler_cells = header["nDopplerCells"]
        n_range_cells   = header["nRangeCells"]

    if version >= 5:
        header["nOutputInterval"]  = u1("i")
        header["nCreateTypeCode"]  = f.read(4).rstrip(b"\x00").decode("ascii", errors="replace")
        header["nCreatorVersion"]  = f.read(4).rstrip(b"\x00").decode("ascii", errors="replace")
        header["nActiveChannels"]  = u1("i")
        header["nSpectraChannels"] = u1("i")
        header["nActiveChanBits"]  = u1("I")
        header["nV5Extent"]        = u1("i")
        n_spectra_channels = header["nSpectraChannels"]

    v6_blocks = {}
    if version >= 6:
        cs6_size  = u1("I")  # nCS6ByteSize — total byte count of all following blocks
        remaining = cs6_size
        while remaining >= 8:
            raw_key = f.read(4)
            if len(raw_key) < 4:
                break
            raw_sz = f.read(4)
            if len(raw_sz) < 4:
                break
            blk_sz = struct.unpack(endian + "I", raw_sz)[0]
            remaining -= 8
            if blk_sz > remaining:
                break
            raw  = f.read(blk_sz)
            remaining -= blk_sz
            bkey = raw_key.decode("ascii", errors="replace")
            parsed = _parse_v6_block(
                bkey, raw, n_range_cells, n_doppler_cells, n_spectra_channels
            )
            if bkey == "TOOL":
                v6_blocks.setdefault("TOOL", []).append(parsed)
            else:
                v6_blocks[bkey] = parsed

    header["_nRangeCells"]      = n_range_cells
    header["_nDopplerCells"]    = n_doppler_cells
    header["_nSpectraChannels"] = n_spectra_channels
    header["_nCsKind"]          = n_cs_kind

    return header, v6_blocks


def _riff_iter(data: bytes, endian: str):
    """Yield (key, key_data) tuples from a RIFF-style byte buffer."""
    pos = 0
    while pos + 8 <= len(data):
        key = data[pos: pos + 4].decode("ascii", errors="replace")
        sz  = struct.unpack_from(endian + "I", data, pos + 4)[0]
        pos += 8
        yield key, data[pos: pos + sz]
        pos += sz


def _read_reduced(f) -> dict:
    raw = f.read()

    magic     = raw[:4]
    big_endian = magic in _REDUCED_MAGIC_BE
    endian    = ">" if big_endian else "<"

    pos = 0

    def next_key():
        nonlocal pos
        key = raw[pos: pos + 4].decode("ascii", errors="replace")
        sz  = struct.unpack_from(endian + "I", raw, pos + 4)[0]
        pos += 8
        kdata = raw[pos: pos + sz]
        pos  += sz
        return key, kdata

    # ── CSSW root ─────────────────────────────────────────────────────────────
    root_key, root_data = next_key()

    head_data = None
    body_data = None

    for key, kdata in _riff_iter(root_data, endian):
        if key == "HEAD":
            head_data = kdata
        elif key == "BODY":
            body_data = kdata

    if head_data is None:
        raise ValueError("Reduced CSS file is missing HEAD block")
    if body_data is None:
        raise ValueError("Reduced CSS file is missing BODY block")

    # ── Parse HEAD ────────────────────────────────────────────────────────────
    header    = {}
    v6_blocks = {}
    db_ref    = -34.2  # default if dbrf absent
    n_rng     = 31
    n_dopp    = 512
    n_sp_ch   = 3

    for key, kdata in _riff_iter(head_data, endian):

        if key == "sign" and len(kdata) >= 208:
            header["sign_nFileVersion"] = kdata[0:4].decode("ascii", errors="replace")
            header["sign_nFileType"]    = kdata[4:8].decode("ascii", errors="replace")
            header["sign_nOwner"]       = kdata[8:12].decode("ascii", errors="replace")
            header["sign_szFileName"]   = kdata[16:80].rstrip(b"\x00").decode("ascii", errors="replace")
            header["sign_szOwnerName"]  = kdata[80:144].rstrip(b"\x00").decode("ascii", errors="replace")
            header["sign_szComment"]    = kdata[144:208].rstrip(b"\x00").decode("ascii", errors="replace")

        elif key == "srcn":
            header["srcn_szSourceFile"] = kdata.decode("ascii", errors="replace")

        elif key == "mcda" and len(kdata) >= 4:
            ts = struct.unpack_from(endian + "I", kdata)[0]
            header["mcda_nDateTime"] = ts
            header["mcda_datetime"]  = _codar_ts(ts)

        elif key == "dbrf" and len(kdata) >= 8:
            db_ref = struct.unpack_from(endian + "d", kdata)[0]
            header["dbrf_fReceiverPowerLossDB"] = db_ref

        elif key == "cs4h":
            cs4h_hdr, cs4h_v6 = _parse_cs4h(kdata, endian)
            header.update(cs4h_hdr)
            v6_blocks.update(cs4h_v6)
            n_rng   = cs4h_hdr.get("_nRangeCells",      n_rng)
            n_dopp  = cs4h_hdr.get("_nDopplerCells",    n_dopp)
            n_sp_ch = cs4h_hdr.get("_nSpectraChannels", n_sp_ch)

        elif key == "alim" and len(kdata) >= 32:
            n_type, n_alim_range = struct.unpack_from(endian + "II", kdata)
            rng_km, bearing      = struct.unpack_from(endian + "ff", kdata, 8)
            first_rng, n_dop_a   = struct.unpack_from(endian + "II", kdata, 16)
            alim_info = {
                "nType": n_type, "nRange": n_alim_range,
                "fRangeKm": rng_km, "fBearingDeg": bearing,
                "nFirstRange": first_rng, "nDopplers": n_dop_a,
            }
            lim_nb = n_alim_range * 16
            if len(kdata) >= 32 + lim_nb:
                lims = np.frombuffer(kdata[32: 32 + lim_nb], dtype=endian + "u4"
                                     ).reshape(n_alim_range, 4).copy()
                alim_info["limits"] = lims
            header["alim"] = alim_info

        elif key == "wlim":
            n_wlim = len(kdata) // 16
            if n_wlim > 0:
                lims = np.frombuffer(kdata[: n_wlim * 16], dtype=endian + "u4"
                                     ).reshape(n_wlim, 4).copy()
                header["wlim"] = {"limits": lims}

    # nFirstRangeCell tells us the offset of the first indx value
    first_range = header.get("nFirstRangeCell", 0)

    # ── Allocate output arrays ────────────────────────────────────────────────
    nan2d = lambda: np.full((n_rng, n_dopp), np.nan, dtype=np.float64)
    self1 = nan2d(); self2 = nan2d(); self3 = nan2d()
    c13r  = nan2d(); c13i  = nan2d()
    c23r  = nan2d(); c23i  = nan2d()
    c12r  = nan2d(); c12i  = nan2d()
    qual  = nan2d()
    # Sign-bit buffers per range cell (set when asgn / csgn key is seen)
    asgn_bufs = [None] * n_rng   # 3 × n_dopp bits for self spectra
    csgn_bufs = [None] * n_rng   # 6 × n_dopp bits for cross spectra

    # Map RIFF data key → (output array, needs_dB_to_power_conversion)
    # Real-world files use cs3a (not ca3a) and real+imag (not magnitude+phase)
    _data_arrays = {
        "cs1a": (self1, True),
        "cs2a": (self2, True),
        "cs3a": (self3, True),
        "ca3a": (self3, True),   # spec uses ca3a; accept both
        "c13r": (c13r,  True),
        "c13i": (c13i,  True),
        "c23r": (c23r,  True),
        "c23i": (c23i,  True),
        "c12r": (c12r,  True),
        "c12i": (c12i,  True),
        # Spec-described magnitude+phase keys (accept if present)
        "c13m": (c13r,  True),   "c13a": (c13i,  False),
        "c23m": (c23r,  True),   "c23a": (c23i,  False),
        "c12m": (c12r,  True),   "c12a": (c12i,  False),
        "csqf": (qual,  False),
    }

    # ── Parse BODY ────────────────────────────────────────────────────────────
    current_indx  = first_range   # raw indx value (1-based if first_range=1)
    current_scal  = None          # (nType, fmin, fmax, fscale)

    for key, kdata in _riff_iter(body_data, endian):

        if key == "indx" and len(kdata) >= 4:
            current_indx = struct.unpack_from(endian + "i", kdata)[0]
            current_scal = None

        elif key == "scal" and len(kdata) >= 16:
            current_scal = struct.unpack_from(endian + "ifff", kdata)

        elif key in _data_arrays and current_scal is not None:
            arr_idx = current_indx - first_range
            if 0 <= arr_idx < n_rng:
                _, fmin, fmax, fscale = current_scal
                raw_u32 = _decode_cssw(kdata, n_dopp, big_endian)
                floats  = _apply_scal(raw_u32, fmin, fmax, fscale)
                arr, is_db = _data_arrays[key]
                arr[arr_idx] = _db_to_power(floats, db_ref) if is_db else floats
            current_scal = None

        elif key == "asgn":
            arr_idx = current_indx - first_range
            if 0 <= arr_idx < n_rng:
                asgn_bufs[arr_idx] = kdata

        elif key == "csgn":
            arr_idx = current_indx - first_range
            if 0 <= arr_idx < n_rng:
                csgn_bufs[arr_idx] = kdata

    # ── Apply self-spectra sign bits (asgn) ───────────────────────────────────
    bpa = (n_dopp + 7) // 8   # bytes per antenna
    for r in range(n_rng):
        sb = asgn_bufs[r]
        if sb is None:
            continue
        for ant_idx, arr in enumerate((self1, self2, self3)):
            off = ant_idx * bpa
            for cell in range(n_dopp):
                bp = off + cell // 8
                if bp < len(sb) and (sb[bp] >> (cell % 8)) & 1:
                    arr[r, cell] = -abs(arr[r, cell])

    # ── Apply cross-spectra sign bits (csgn) ──────────────────────────────────
    # csgn packs 6 arrays (c13r,c13i,c23r,c23i,c12r,c12i) each n_dopp bits
    cross_arrs = (c13r, c13i, c23r, c23i, c12r, c12i)
    for r in range(n_rng):
        sb = csgn_bufs[r]
        if sb is None:
            continue
        for comp_idx, arr in enumerate(cross_arrs):
            off = comp_idx * bpa
            for cell in range(n_dopp):
                bp = off + cell // 8
                if bp < len(sb) and (sb[bp] >> (cell % 8)) & 1:
                    arr[r, cell] = -abs(arr[r, cell])

    # ── Reconstruct complex cross spectra from real + imaginary parts ─────────
    cross12 = (c12r + 1j * c12i).astype(np.complex128)
    cross13 = (c13r + 1j * c13i).astype(np.complex128)
    cross23 = (c23r + 1j * c23i).astype(np.complex128)

    _drop_private(header)

    return {
        "file_type": "reduced",
        "header":    header,
        "v6_blocks": v6_blocks,
        "data": {
            "antenna1_self": self1.astype(np.float32),
            "antenna2_self": self2.astype(np.float32),
            "antenna3_self": self3.astype(np.float32),
            "cross_12":      cross12,
            "cross_13":      cross13,
            "cross_23":      cross23,
            "quality":       qual.astype(np.float32),
        },
    }
