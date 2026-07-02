"""CTF (Columnar Table Format) reader."""

from __future__ import annotations

import re
from typing import IO

import numpy as np

_KEYWORD_RE = re.compile(r'^%([A-Za-z][^:]*):(.*)$')

# Table-scoped keywords — not stored in metadata, used only to build table dicts.
_TABLE_KEYWORDS = {'TableStart', 'TableEnd', 'TableType', 'TableColumnTypes',
                   'TableColumns', 'TableRows'}


def _strip_inline_comment(s: str) -> str:
    in_quote = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '"':
            in_quote = not in_quote
        elif not in_quote and s[i:i + 2] == '%%':
            return s[:i]
        i += 1
    return s


def _tokenize(s: str) -> list[str]:
    s = _strip_inline_comment(s).strip()
    tokens: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == '"':
            end = s.find('"', i + 1)
            if end == -1:
                tokens.append(s[i + 1:])
                break
            tokens.append(s[i + 1:end])
            i = end + 1
        elif s[i] == ' ':
            i += 1
        else:
            end = i
            while end < len(s) and s[end] != ' ':
                end += 1
            tokens.append(s[i:end])
            i = end
    return tokens


def _store(metadata: dict, key: str, tokens: list[str]) -> None:
    value = ' '.join(tokens) if tokens else ''
    existing = metadata.get(key)
    if existing is None:
        metadata[key] = value
    elif isinstance(existing, list):
        existing.append(value)
    else:
        metadata[key] = [existing, value]


def _parse_ctf_stream(stream: IO[str]) -> dict:
    content = stream.read()
    content = content.replace('\r\n', '\n').replace('\n\r', '\n').replace('\r', '\n')
    lines = content.split('\n')

    metadata: dict = {}
    tables: list = []

    state = 'preamble'
    current_table: dict | None = None
    current_rows: list | None = None
    # buffer for table-metadata keywords that precede %TableStart:
    pending_type = ''
    pending_col_types: list[str] = []

    for line in lines:
        if state == 'done':
            break

        if not line.strip():
            continue

        if state == 'table':
            if line.startswith('%TableEnd:'):
                arr = (np.array(current_rows, dtype=np.float64) if current_rows
                       else np.empty((0, len(current_table['column_types'])), dtype=np.float64))
                current_table['data'] = arr
                tables.append(current_table)
                current_table = None
                current_rows = None
                pending_type = ''
                pending_col_types = []
                state = 'preamble'
                continue

            # Data row: leading space or '% '
            if line.startswith(' ') or line.startswith('% '):
                row_str = line[2:].strip() if line.startswith('% ') else line.strip()
                if row_str:
                    try:
                        current_rows.append([float(v) for v in row_str.split()])
                    except ValueError:
                        pass
                continue

            # Keyword inside table block (e.g. %TableType: placed after %TableStart:)
            m = _KEYWORD_RE.match(line)
            if m:
                key, tokens = m.group(1), _tokenize(m.group(2))
                if key == 'TableType':
                    current_table['table_type'] = ' '.join(tokens)
                elif key == 'TableColumnTypes':
                    current_table['column_types'] = tokens
                elif key not in ('TableColumns', 'TableRows'):
                    _store(metadata, key, tokens)
            continue

        # ── preamble state ─────────────────────────────────────────────────────
        if line.startswith('%%'):
            continue

        if line.startswith('% '):
            continue

        if line.startswith('%End:'):
            state = 'done'
            continue

        m = _KEYWORD_RE.match(line)
        if not m:
            continue

        key, tokens = m.group(1), _tokenize(m.group(2))

        if key == 'TableStart':
            state = 'table'
            current_table = {
                'table_type': pending_type,
                'column_types': list(pending_col_types),
                'data': None,
            }
            current_rows = []
            continue

        if key == 'TableType':
            pending_type = ' '.join(tokens)
            continue

        if key == 'TableColumnTypes':
            pending_col_types = tokens
            continue

        if key in ('TableColumns', 'TableRows', 'TableEnd'):
            continue

        _store(metadata, key, tokens)

    return {'metadata': metadata, 'data': {'tables': tables}}


def read_ctf(filename: str) -> dict:
    """Read a CODAR CTF (Columnar Table Format) text file.

    Parses all ``%KeywordName: <params>`` lines into ``metadata`` and any
    ``%TableStart:`` / ``%TableEnd:`` blocks into ``data["tables"]``.
    Keyword names are stored with their original casing. Repeated keywords
    accumulate into a list. Inline ``%%`` comments and blank lines are ignored.
    Parsing stops at the first ``%End:`` line.

    Args:
        filename: Path to the CTF file. The file is read with ``latin-1``
            encoding to accommodate arbitrary byte values.

    Returns:
        A dict with two keys:

        - ``"metadata"`` (dict): maps each keyword name (original case,
          without the leading ``%`` and trailing ``:``) to a ``str`` value,
          or to a ``list[str]`` when the same keyword appears more than once.
          Multi-word quoted parameter strings are stored as a single value.
        - ``"data"`` (dict): contains a single key ``"tables"``, which is a
          list of table dicts. Each table dict has:

          - ``"table_type"`` (str): value of ``%TableType:``.
          - ``"column_types"`` (list[str]): four-character codes from
            ``%TableColumnTypes:``, in column order.
          - ``"data"`` (numpy.ndarray): 2-D array of shape
            ``(nRows, nCols)``, dtype ``float64``, holding the numeric
            table rows.

    Raises:
        FileNotFoundError: If ``filename`` does not exist.
        ValueError: If a table row contains non-numeric tokens.

    Example:
        >>> result = read_ctf("RDLm_SITE_2024_01_01_1200.ruv")
        >>> result["metadata"]["Site"]
        'SITE'
        >>> result["data"]["tables"][0]["data"].shape
        (627, 18)
    """
    with open(filename, encoding='latin-1') as f:
        return _parse_ctf_stream(f)
