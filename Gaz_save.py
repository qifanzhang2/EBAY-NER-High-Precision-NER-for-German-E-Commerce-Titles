
import os
import re
import json
import csv
import unicodedata
from collections import defaultdict, Counter

import argparse
import pandas as pd

# ----------------------- Canonicalization -------------------------
def canonicalize_for_match(s: str) -> str:
    """Match the NER canonicalizer (NFKD, ß->ss, ASCII fold, dash unification, comma->dot)."""
    x = (s or "").strip()
    x = unicodedata.normalize("NFKD", x)
    x = x.replace("ß", "ss")
    x = x.encode("ascii", "ignore").decode("ascii")
    x = x.replace("\u2013","-").replace("\u2014","-").replace("–","-").replace("—","-")
    x = x.replace(",", ".")
    x = re.sub(r"\s+", " ", x)
    return x.lower()

# ----------------------- Aspects / Categories ---------------------
ASPECT_TO_CATS = {
    "Anwendung": [2],
    "Anzahl_Der_Einheiten": [1, 2],
    "Besonderheiten": [1, 2],
    "Breite": [2],
    "Bremsscheiben-Aussendurchmesser": [1],
    "Bremsscheibenart": [1],
    "Einbauposition": [1, 2],
    "Farbe": [1],
    "Größe": [1, 2],
    "Hersteller": [1, 2],
    "Herstellernummer": [1, 2],
    "Herstellungsland_Und_-Region": [1],
    "Im_Lieferumfang_Enthalten": [1, 2],
    "Kompatible_Fahrzeug_Marke": [1, 2],
    "Kompatibles_Fahrzeug_Jahr": [1, 2],
    "Kompatibles_Fahrzeug_Modell": [1, 2],
    "Länge": [2],
    "Material": [1],
    "Maßeinheit": [1, 2],
    "Menge": [2],
    "Modell": [1, 2],
    "Oberflächenbeschaffenheit": [1],
    "Oe/Oem_Referenznummer(N)": [1, 2],
    "Produktart": [1, 2],
    "Produktlinie": [1],
    "SAE_Viskosität": [2],
    "Stärke": [1],
    "Technologie": [1],
    "Zähnezahl": [2]
}
ALL_ASP = sorted(ASPECT_TO_CATS.keys())

# ----------------------- Essential validators ---------------------
OE_PATTERN_MIXED = re.compile(r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9._/\-]{5,}$')
OE_PATTERN_NUM   = re.compile(r'^\d{5,}$')
YEAR_PATTERNS = [
    re.compile(r'^(19|20)\d{2}$'),
    re.compile(r'^\d{2}-\d{2}$'), re.compile(r'^\d{4}-\d{2}$'),
    re.compile(r'^\d{2}-\d{4}$'), re.compile(r'^\d{4}-\d{4}$')
]
EINBAU_WHITELIST = {
    "vorne","hinten","links","rechts","vorderachse","hinterachse","va","ha",
    "fahrerseite","beifahrerseite","vorn","hint"
}
ROTOR_MIN, ROTOR_MAX = 200, 420  # mm

def is_valid_minimal(asp: str, val: str) -> bool:
    """Minimal precision guards; everything else passes to maximize coverage."""
    v = (val or "").strip()
    if not v:
        return False
    v_can = canonicalize_for_match(v)

    if asp in {"Herstellernummer","Oe/Oem_Referenznummer(N)"}:
        return bool(OE_PATTERN_NUM.match(v) or OE_PATTERN_MIXED.match(v))
    if asp == "Einbauposition":
        v0 = v_can.replace("-", "")
        return v0 in {w.replace("-", "") for w in EINBAU_WHITELIST}
    if asp == "Bremsscheiben-Aussendurchmesser":
        m = re.search(r'(\d{2,4})\s*mm', v_can) or re.search(r'^\s*(\d{2,4})\s*$', v_can)
        return bool(m) and (ROTOR_MIN <= int(m.group(1)) <= ROTOR_MAX)
    if asp == "Kompatibles_Fahrzeug_Jahr":
        return any(p.match(v) for p in YEAR_PATTERNS)
    # Basic sanity: not only punctuation, reasonable length
    if len(re.sub(r"[A-Za-z0-9]", "", v_can)) >= len(v_can) - 1:
        return False
    return 1 <= len(v_can) <= 64

# Generic singletons we keep out of HARD for Produktart/Lieferumfang (unless seen in train)
GENERIC_ONEWORD = {"kit","set","satz","kpl","komplett"}

# ------------------ BIO decode for training spans -----------------
def decode_train_spans_for_record(tokens, raw_tags):
    # Train format: O / "" (I-cont) / "<AspectName>" (B-aspect)
    bio = []
    prev = None
    for t in raw_tags:
        if t == "O":
            bio.append(("O", None)); prev=None
        elif t == "":
            bio.append(("I", prev) if prev else ("O", None))
        else:
            bio.append(("B", t)); prev=t
    spans = []
    i, n = 0, min(len(tokens), len(bio))
    while i < n:
        tag, asp = bio[i]
        if tag == "B":
            j = i + 1
            while j < n and bio[j] == ("I", asp):
                j += 1
            text = " ".join(tokens[i:j]).strip()
            if asp and text:
                spans.append((asp, text))
            i = j
        else:
            i += 1
    return spans

# -------------------- Mining from training ------------------------
def mine_from_train(train_tsv: str):
    df = pd.read_csv(train_tsv, sep="\t", quoting=csv.QUOTE_NONE, engine="python", keep_default_na=False)
    rec_groups = df.groupby(df.columns[0])  # "Record Number"
    per_aspect_counts = {a: Counter() for a in ALL_ASP}
    per_aspect_forms  = {a: defaultdict(int) for a in ALL_ASP}  # canonical -> representative count
    total_gold = Counter()

    for _, g in rec_groups:
        tokens = g.iloc[:,3].astype(str).tolist()
        tags   = g.iloc[:,4].astype(str).tolist()
        spans = decode_train_spans_for_record(tokens, tags)
        for asp, text in spans:
            if asp not in ALL_ASP:
                continue
            total_gold[asp] += 1
            if not is_valid_minimal(asp, text):
                continue
            can = canonicalize_for_match(text)
            per_aspect_counts[asp][can] += 1
            per_aspect_forms[asp][text] += 1

    # pick most frequent surface form per canonical
    bestform = {a:{} for a in ALL_ASP}
    for asp in ALL_ASP:
        cand = defaultdict(lambda: Counter())
        for orig, c in per_aspect_forms[asp].items():
            cand[canonicalize_for_match(orig)][orig] += c
        for can, ctr in cand.items():
            bestform[asp][can] = ctr.most_common(1)[0][0]
    return per_aspect_counts, bestform, total_gold

# ----------------- Build HARD/SOFT (coverage-first) ---------------
def build_gazetteers(per_aspect_counts, per_aspect_bestform):
    HARD = {a: [] for a in ALL_ASP}
    SOFT = {a: [] for a in ALL_ASP}

    for asp in ALL_ASP:
        ctr = per_aspect_counts[asp]       # canonical -> freq in train
        best = per_aspect_bestform[asp]    # canonical -> best surface form
        for can, freq in ctr.items():
            orig = best.get(can, can)
            if not is_valid_minimal(asp, orig):
                continue

            v_can = canonicalize_for_match(orig)
            toks = v_can.split()

            # if single-token generic and NEVER seen more than once in train, push to SOFT
            if asp in {"Produktart","Im_Lieferumfang_Enthalten"} and len(toks)==1:
                if toks[0] in GENERIC_ONEWORD and freq < 2:
                    SOFT[asp].append(orig)
                    continue

            HARD[asp].append(orig)

    # De-dup per aspect by canonical (keep longest surface)
    def dedup_longest(d):
        out = {}
        for v in d:
            c = canonicalize_for_match(v)
            if c not in out or len(v) > len(out[c]):
                out[c] = v
        return sorted(out.values(), key=lambda s: (len(s), s))

    for a in ALL_ASP:
        HARD[a] = dedup_longest(HARD[a])
        # For coverage-first behavior, you may simply set SOFT[a] = []
        SOFT[a] = dedup_longest(SOFT[a])

    return HARD, SOFT

# --------------------------- Coverage -----------------------------
def compute_coverage(train_tsv: str, hard_gaz):
    df = pd.read_csv(train_tsv, sep="\t", quoting=csv.QUOTE_NONE, engine="python", keep_default_na=False)
    rec_groups = df.groupby(df.columns[0])
    total_by_asp = Counter()
    hit_by_asp   = Counter()

    GAZ_CAN = {a: {canonicalize_for_match(v) for v in vals} for a, vals in hard_gaz.items()}

    for _, g in rec_groups:
        tokens = g.iloc[:,3].astype(str).tolist()
        tags   = g.iloc[:,4].astype(str).tolist()
        spans  = decode_train_spans_for_record(tokens, tags)
        for asp, text in spans:
            if asp not in ALL_ASP:
                continue
            total_by_asp[asp] += 1
            if canonicalize_for_match(text) in GAZ_CAN.get(asp, set()):
                hit_by_asp[asp] += 1

    report = {
        "total_gold": dict(total_by_asp),
        "hits_in_hard": dict(hit_by_asp),
        "per_aspect_coverage": {
            a: (hit_by_asp[a] / total_by_asp[a] if total_by_asp[a] else 0.0) for a in ALL_ASP
        },
        "overall_coverage": (
            sum(hit_by_asp.values()) / max(1, sum(total_by_asp.values()))
        )
    }
    return report

# ------------------------------ Main ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="Tagged_Titles_Train.tsv")
    ap.add_argument("--out-hard", default="gazetteer_hard.json")
    ap.add_argument("--out-soft", default="gazetteer_soft.json")
    ap.add_argument("--out-main", default="gazetteer.json")
    ap.add_argument("--out-cover", default="gazetteer_coverage.json")
    args = ap.parse_args()

    if not os.path.exists(args.train):
        raise FileNotFoundError(f"Training file not found: {args.train}")

    print("➤ Mining labeled spans from training (coverage-first)...")
    per_aspect_counts, per_aspect_bestform, total_gold = mine_from_train(args.train)

    print("➤ Building HARD/SOFT gazetteers (no DF pruning, minimal guards)...")
    HARD, SOFT = build_gazetteers(per_aspect_counts, per_aspect_bestform)

    with open(args.out_hard, "w", encoding="utf-8") as f:
        json.dump(HARD, f, ensure_ascii=False, indent=2)
    with open(args.out_soft, "w", encoding="utf-8") as f:
        json.dump(SOFT, f, ensure_ascii=False, indent=2)
    with open(args.out_main, "w", encoding="utf-8") as f:
        json.dump(HARD, f, ensure_ascii=False, indent=2)

    print(f"✔ Wrote {args.out_hard} (coverage-first HARD)")
    print(f"✔ Wrote {args.out_soft} (SOFT)")
    print(f"✔ Wrote {args.out_main} (alias of HARD)")

    print("➤ Computing training coverage for HARD gazetteer...")
    cover = compute_coverage(args.train, HARD)
    with open(args.out_cover, "w", encoding="utf-8") as f:
        json.dump(cover, f, ensure_ascii=False, indent=2)

    # Console summary
    total_items = {a: len(HARD[a]) for a in ALL_ASP}
    print("── Gazetteer size (HARD) per aspect:")
    for a in sorted(ALL_ASP):
        print(f"  {a:32s}  {total_items[a]:6d}")
    print(f"Overall HARD items: {sum(total_items.values()):,}")
    print("── Coverage (HARD) on train:")
    print(f"  Overall coverage: {cover['overall_coverage']:.3f}")

if __name__ == "__main__":
    main()
