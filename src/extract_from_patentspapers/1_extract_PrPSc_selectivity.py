print("✅ 01b_extract_PrPSc_selectivity.py started")

import re
import sys
from pathlib import Path

import pdfplumber
import pandas as pd

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "XAA": "X",
}

FIXES = {
    "GLIN": "GLN",
    "TEU": "LEU",
    "TELU": "LEU",
    "LIEU": "LEU",
    "LIEU.": "LEU",
    "WALL": "VAL",
    "THIR": "THR",
    "LLYS": "LYS",
    "LLY": "LYS",
}


def normalize_token(tok: str) -> str:
    t = re.sub(r"[^A-Za-z]", "", tok)
    if not t:
        return ""
    t = t.upper()
    return FIXES.get(t, t)


def token_to_aa1(tok: str) -> str | None:
    t = normalize_token(tok)
    if not t:
        return None
    return AA3_TO_1.get(t)


def extract_full_text(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def iter_seqid_blocks(full_text: str):
    """
    Yields (seq_id:int, block_text:str) for each "(2) INFORMATION FOR SEQ ID NO: X" block.
    """
    pattern = re.compile(
        r"\(2\)\s+INFORMATION FOR SEQ ID NO:\s*([0-9]+)\s*:?(.*?)(?=\(2\)\s+INFORMATION FOR SEQ ID NO:|\Z)",
        re.S,
    )
    for seq_id, block in pattern.findall(full_text):
        yield int(seq_id), block


def block_declared_length(block: str) -> int | None:
    m = re.search(r"LENGTH:\s*([0-9]+)\s*amino acids", block, flags=re.I)
    return int(m.group(1)) if m else None


def block_is_peptide(block: str) -> bool:
    return bool(re.search(r"MOLECULE TYPE:\s*peptide", block, flags=re.I))


def block_sequence_text(block: str) -> str:
    """
    Tries to grab the sequence tokens after the SEQUENCE DESCRIPTION header.
    If that fails, uses the whole block.
    """
    m = re.search(
        r"SEQUENCE DESCRIPTION:.*?SEQ ID NO[:\s]*[0-9]+\s*:?(.*)",
        block,
        flags=re.I | re.S,
    )
    return m.group(1) if m else block


def tokens_to_sequence(seq_text: str) -> str:
    toks = re.split(r"\s+", seq_text)
    aas = []
    for tok in toks:
        aa = token_to_aa1(tok)
        if aa:
            aas.append(aa)
    return "".join(aas)


def guess_chain_and_cdr3(seq: str):
    """
    Heuristic CDR3 extraction:
      - heavy-like: C {3..40} W(G/Q)
      - light-like: C {3..40} F(G/Q)
    Returns (chain_guess, cdr3, method) or (None, None, None).
    """
    m = re.search(r"C([A-Z]{3,40})W[GQ]", seq)
    if m:
        return "heavy_like", m.group(1), "C...W(G/Q)"
    m = re.search(r"C([A-Z]{3,40})F[GQ]", seq)
    if m:
        return "light_like", m.group(1), "C...F(G/Q)"
    return None, None, None


def likely_antibody_variable_region(seq: str) -> bool:
    """
    Kept from your original script. Not used as a hard filter here.
    """
    starts = (
        "QVQL",
        "EVQL",
        "DVQL",
        "DIQM",
        "DVVM",
        "EIVL",
        "QIVL",
        "QSVL",
        "ELVL",
    )
    return seq.startswith(starts)


def has_long_repeat(seq: str, max_run: int = 6) -> bool:
    """
    Low-complexity filter:
    Flags sequences with runs of the same amino acid longer than max_run+1.
    Example (max_run=6): matches 7+ repeats like GGGGGGG or PPPPPPP.
    """
    if not isinstance(seq, str):
        return False
    return bool(re.search(r"(.)\1{" + str(max_run) + r",}", seq))


def main():
    if len(sys.argv) < 2:
        print("Usage: python 01b_extract_PrPSc_selectivity.py <path_to_pdf>")
        sys.exit(1)
    pdf_path = Path(sys.argv[1]).expanduser().resolve()
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)
    full_text = extract_full_text(pdf_path)
    rows = []
    for seq_id, block in iter_seqid_blocks(full_text):
        if not block_is_peptide(block):
            continue
        declared_len = block_declared_length(block)
        seq_text = block_sequence_text(block)
        seq_1letter = tokens_to_sequence(seq_text)
        chain_guess, cdr3, cdr3_method = guess_chain_and_cdr3(seq_1letter)
        rows.append(
            {
                "seq_id": seq_id,
                "declared_len": declared_len,
                "extracted_len": len(seq_1letter),
                "likely_antibody_var": likely_antibody_variable_region(seq_1letter),
                "chain_guess": chain_guess,
                "cdr3": cdr3,
                "cdr3_method": cdr3_method,
                "sequence_aa": seq_1letter,
            }
        )
    df = pd.DataFrame(rows).sort_values(
        ["likely_antibody_var", "seq_id"], ascending=[False, True]
    )
    df["cdr3_len"] = df["cdr3"].astype("string").str.len()
    df = df[
        df["cdr3"].notna() & (df["cdr3_len"] >= 8) & (df["cdr3_len"] <= 30)
    ].reset_index(drop=True)
    df = df[~df["sequence_aa"].apply(has_long_repeat)].reset_index(drop=True)
    out_name = f"cdr3_extracted_{pdf_path.stem}.csv"
    df.to_csv(out_name, index=False)
    hits = df[
        [
            "seq_id",
            "chain_guess",
            "cdr3",
            "cdr3_len",
            "declared_len",
            "extracted_len",
            "likely_antibody_var",
        ]
    ]
    print(f"\nSaved: {out_name}")
    print(f"Total peptide SEQ IDs kept after filters: {len(df)}")
    print(f"CDR3 detected (heuristic, after filters): {len(hits)}\n")
    if len(hits) > 0:
        print(hits.to_string(index=False))
    else:
        print(
            "No CDR3-like motifs found after filters. If the PDF is scanned, you may need OCR."
        )


if __name__ == "__main__":
    main()
