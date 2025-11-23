
import os, re, json, csv, random, math, unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torchcrf import CRF

# ============================= Config =============================
SEED  = 42
MODEL_NAME = "xlm-roberta-large"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
LR = 2e-5
EPOCHS = 6
EARLY_STOP_PATIENCE = 2
N_FOLDS = 5

CKPT_DIR = "checkpoints_proplus"
os.makedirs(CKPT_DIR, exist_ok=True)

TRAIN_TSV = os.environ.get("TRAIN_TSV", "Tagged_Titles_Train.tsv")
TITLES_TSV = os.environ.get("TITLES_TSV", "Listing_Titles.tsv")
GAZ_FILE   = os.environ.get("GAZ_FILE", "gazetteer_hard.json")
GAZ_SOFT_FILE = os.environ.get("GAZ_SOFT_FILE", "gazetteer_soft.json")
SUBMIT_OUT = os.environ.get("SUBMIT_OUT", "predictions.tsv")

RUN_STRICT_CV = os.environ.get("STRICT_CV", "0") == "1"
USE_STAGE2_VERIFIER = os.environ.get("USE_VERIFIER", "1") == "1"

USE_CHARCNN = os.environ.get("USE_CHARCNN", "1") == "1"
USE_ASP_GAZ = os.environ.get("USE_ASP_GAZ", "1") == "1"
CHAR_MAXLEN = int(os.environ.get("CHAR_MAXLEN", 24))

MAX_PER_TITLE_BASE = {
    1: defaultdict(lambda: 3, **{
        "Hersteller": 1, 
        "Produktart": 1, 
        "Einbauposition": 2,
        "Kompatible_Fahrzeug_Marke": 3, 
        "Kompatibles_Fahrzeug_Modell": 3,
        "Im_Lieferumfang_Enthalten": 2,  # Reduced from 3 to 2
        "Herstellernummer": 3, 
        "Oe/Oem_Referenznummer(N)": 3,
        "Bremsscheiben-Aussendurchmesser": 2,
        "Anzahl_Der_Einheiten": 2,
    }),
    2: defaultdict(lambda: 3, **{
        "Hersteller": 1,
        "Produktart": 1,
        "Kompatible_Fahrzeug_Marke": 3,
        "Kompatibles_Fahrzeug_Modell": 3,
        "Im_Lieferumfang_Enthalten": 2,  # Also strict for category 2
        "Herstellernummer": 3,
    })
}

AGGR_NO_GAZ_RELAX = {
    "Produktart", "Kompatibles_Fahrzeug_Modell", "Kompatible_Fahrzeug_Marke",
    "Im_Lieferumfang_Enthalten", "Hersteller"
}

ASPECT_TO_CATS = {
    "Anwendung": [2],
    "Anzahl_Der_Einheiten": [1, 2],
    "Besonderheiten": [1, 2],
    "Breite": [2],
    "Bremsscheiben-Aussendurchmesser": [1],
    "Bremsscheibenart": [1],
    "Einbauposition": [1, 2],
    "Farbe": [1],
    "Größe": [1, 2],  # Fixed: was GrÃ¶ÃŸe
    "Hersteller": [1, 2],
    "Herstellernummer": [1, 2],
    "Herstellungsland_Und_-Region": [1],
    "Im_Lieferumfang_Enthalten": [1, 2],
    "Kompatible_Fahrzeug_Marke": [1, 2],
    "Kompatibles_Fahrzeug_Jahr": [1, 2],
    "Kompatibles_Fahrzeug_Modell": [1, 2],
    "Länge": [2],  # Fixed: was LÃ¤nge
    "Material": [1],
    "Maßeinheit": [1, 2],  # Fixed: was MaÃŸeinheit
    "Menge": [2],
    "Modell": [1, 2],
    "Oberflächenbeschaffenheit": [1],
    "Oe/Oem_Referenznummer(N)": [1, 2],
    "Produktart": [1, 2],
    "Produktlinie": [1],
    "SAE_Viskosität": [2],  # Fixed: was SAE_ViskositÃ¤t
    "Stärke": [1],  # Fixed: was StÃ¤rke
    "Technologie": [1],
    "Zähnezahl": [2]  # Fixed: was ZÃ¤hnezahl
}
ALLOWED_ASP = {
    1: {a for a,c in ASPECT_TO_CATS.items() if 1 in c},
    2: {a for a,c in ASPECT_TO_CATS.items() if 2 in c},
}
ALL_ASP = sorted(ASPECT_TO_CATS.keys())

# Stage-1 seed thresholds (same spirit; OOF tuner adjusts)
VOTE_THRESH_5 = defaultdict(lambda: 3, **{
    "Besonderheiten": 3,
    "Im_Lieferumfang_Enthalten": 3,
    "Oe/Oem_Referenznummer(N)": 3,
    "Herstellernummer": 4,
    "Hersteller": 4,
    "Produktart": 5,
    "Kompatible_Fahrzeug_Marke": 4,
    "Kompatibles_Fahrzeug_Modell": 3,
    "Einbauposition": 3,
    "Bremsscheiben-Aussendurchmesser": 2,
})
CONF_THRESH = defaultdict(lambda: 0.55, **{
    "Besonderheiten": 0.80,
    "Im_Lieferumfang_Enthalten": 0.85,
    "Oe/Oem_Referenznummer(N)": 0.70,
    "Herstellernummer": 0.78,
    "Hersteller": 0.75,
    "Produktart": 0.82,
    "Kompatible_Fahrzeug_Marke": 0.78,
    "Kompatibles_Fahrzeug_Modell": 0.78,
    "Einbauposition": 0.70,
    "Bremsscheiben-Aussendurchmesser": 0.60,
})
CONF_THRESH_STRICT = dict(CONF_THRESH)
CONF_THRESH_STRICT.update({
    "Im_Lieferumfang_Enthalten": 0.90, 
    "Kompatible_Fahrzeug_Marke": 0.83, 
    "Produktart": 0.85 
})

VOTE_THRESH_5_STRICT = dict(VOTE_THRESH_5)
VOTE_THRESH_5_STRICT.update({
    "Im_Lieferumfang_Enthalten": 4,
    "Kompatible_Fahrzeug_Marke": 4,
    "Produktart": 4
})

# Lower thresholds for rare aspects stay as they are:
for rare_aspect in ["Länge", "Breite", "Farbe", "Zähnezahl", "Technologie", "Herstellungsland_Und_-Region"]:
    CONF_THRESH_STRICT[rare_aspect] = 0.40
    VOTE_THRESH_5_STRICT[rare_aspect] = 2


# Conservative step sizes to avoid over-shooting
CALIB_CONF_STEPS = (0.10, 0.05, 0.02)
CALIB_VOTE_STEPS = (0.10, 0.05)

CALIB_ITERS = 4


# Contest safeguard
CALIB_RECALL_FLOOR_CAT = {1: 0.45, 2: 0.45}

# Stage-2 verifier
VER_RECALL_FLOOR = 0.45
VER_QUANT_STEPS = 61
VER_SAFE_FLOOR  = 0.905

PN_ASPECTS = {"Herstellernummer", "Oe/Oem_Referenznummer(N)"}
ROTOR_MIN, ROTOR_MAX = 200, 420  # mm
# ---- Precision pack knobs ----
HIGH_PRECISION = os.environ.get("HIGH_PRECISION", "1") == "1"

CALIB_PREC_FLOOR_DROP = 0.006 if HIGH_PRECISION else 0.002  # relative to START micro-P
VER_PREC_FLOOR = float(os.environ.get("VER_PREC_FLOOR", 0.997 if HIGH_PRECISION else 0.990))

# KFM guards
KFM_MAX_LEN = 12
KFM_MAX_DIGIT_RATIO = 0.40
KFM_NEAR_BRAND_WIN = 5
KFM_REQUIRE_MODELISH = True

# Tokens/patterns that should not constitute a 'model'
KFM_BAN_TOK = {
    "kit","set","universal","universal-fit","universalfit",
    "front","rear","left","right","links","rechts","va","ha",
    "st","stueck","stück","paar","mm","cm","ml","l","satz","seta","paarweise","qualitat","qualität","neu","original"

}
# “Model-ish” patterns: E46, W203, F30, 8P, MK4, B8, etc.
KFM_ALLOW_PATTERNS = [
    r"^[efgw][0-9]{2}$", r"^[abdefghjkmoprstvwz]\d$", r"^mk\d{1,2}$",
    r"^\d{1,2}[a-z]$", r"^[a-z]\d{1,3}[a-z]?$", r"^typ\s?\w+$",
    r"^[ivx]{2,4}$"    # was {1,4}: now we disallow bare 'v', 'i', 'x' as "model-ish"
]
KFM_ALLOW_RE = [re.compile(p, re.I) for p in KFM_ALLOW_PATTERNS]

# Hersteller disambig context (extends 'fÃ¼r/fuer')
FUER_SYNONYMS = {"für","fuer","fur","passend","passend_fuer","geeignet","kompatibel","compatible","fits","for"}

# PN hygiene
VOLT_RE = re.compile(r"^\s*(12|24)\s*v\s*$", re.I)
PN_NUM_LEN_MIN, PN_NUM_LEN_MAX = 6, 18
# ---- Produktart high-precision guards ----
# Tokens that are too generic to stand alone as Produktart
PA_GENERIC_SOLO = {
    "kit","set","satz","komplett","komplettsatz",
    "paar","stück","stuck","universal","qualität","qualitat",
    "neu","original"
}
# Words that are frequently context noise for Produktart
PA_GENERIC_NOISE = {
    "front","rear","links","rechts","va","ha",
    "vorderachse","hinterachse","fahrerseite","beifahrerseite"
}

# Domain "head" nouns that make a Produktart specific enough
PA_HEADS = {
    1: {  # Category 1: brake kits
        "bremsscheibe","bremsscheiben","bremsscheibensatz",
        "bremsbelag","bremsbeläge","bremsbelagsatz","belag","beläge",  # Fixed encoding
        "bremssattel","bremstrommel","verschleißanzeiger","verschleissanzeiger",
        "scheibenbremsbelag","abs-ring","absring"
    },
    2: {  # Category 2: engine timing kits
        "zahnriemen","zahnriemensatz","riemen","riemensatz",
        "steuerkette","steuerketten","steuerkettensatz",
        "wasserpumpe","wapu","spannrolle","umlenkrolle",
        "kettenspanner","kettenschiene","kettenrad","steuerradsatz"
    }
}

# Cross-category blocks: if Produktart mentions these, it's almost surely the other category
PA_CROSS_BAN = {
    1: {"zahnriemen","riemen","steuerkette","steuerketten","wasserpumpe"},      # ban timing terms in brakes
    2: {"bremse","bremsscheibe","bremsscheiben","bremsbelag","belag","sattel"}  # ban brake terms in timing
}

# ============================ GPU/Amp =============================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
USE_COMPILE = hasattr(torch, "compile") and os.environ.get("TORCH_COMPILE","0") == "1"

# ============================ Utils ===============================
def canonicalize_for_match(s: str) -> str:
    x = (s or "").strip()
    x = unicodedata.normalize("NFKD", x)
    x = x.replace("ß", "ss")
    x = x.encode("ascii", "ignore").decode("ascii")
    x = x.replace("\u2013","-").replace("\u2014","-")  # Remove corrupted em-dash replacements
    x = x.replace(",", ".")
    x = re.sub(r"\s+", " ", x)
    return x.lower()
def canonicalize_output(s: str) -> str:
    return s.strip()

# Gazetteer loaders
def _try_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_gazetteers(hard_path=GAZ_FILE, soft_path=GAZ_SOFT_FILE):
    def _try(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    hard_candidates = [hard_path, "gazetteer_hard (2).json", "gazetteer_hard(2).json", "gazetteer_hard.json"]
    g_hard = None
    for p in hard_candidates:
        if g_hard is None:
            g_hard = _try(p)
    if g_hard is None:
        raise FileNotFoundError("Missing hard gazetteer (tried: %s)" % hard_candidates)

    soft_candidates = [soft_path, "gazetteer_soft (2).json", "gazetteer_soft(2).json", "gazetteer_soft.json"]
    g_soft = None
    for p in soft_candidates:
        if g_soft is None:
            g_soft = _try(p)
    return g_hard, (g_soft or {})

GAZ_ORIG_HARD, GAZ_ORIG_SOFT = load_gazetteers()

def build_gaz_sets(g_orig):
    val_set = set()
    tok_set = set()
    per_asp_val = {asp: set() for asp in g_orig}
    per_asp_tok = {asp: set() for asp in g_orig}
    per_asp_prefix = {asp: set() for asp in g_orig}
    for asp, vals in g_orig.items():
        for v in vals:
            vc = canonicalize_for_match(v)
            val_set.add(vc)
            per_asp_val[asp].add(vc)
            toks = re.split(r"[ \-_/]", vc)
            toks = [t.strip() for t in toks if t.strip()]
            for t in toks:
                if any(ch.isalpha() for ch in t) or len(t) >= 3:
                    tok_set.add(t)
                    per_asp_tok[asp].add(t)
            if toks:
                per_asp_prefix[asp].add(toks[0])
    return val_set, tok_set, per_asp_val, per_asp_tok, per_asp_prefix

GAZ_VAL_HARD, GAZ_TOK_HARD, GAZ_CANON_HARD, GAZ_TOK_ASP_HARD, GAZ_PREF_ASP_HARD = build_gaz_sets(GAZ_ORIG_HARD)
if GAZ_ORIG_SOFT:
    GAZ_VAL_SOFT, GAZ_TOK_SOFT, GAZ_CANON_SOFT, GAZ_TOK_ASP_SOFT, GAZ_PREF_ASP_SOFT = build_gaz_sets(GAZ_ORIG_SOFT)
else:
    GAZ_VAL_SOFT, GAZ_TOK_SOFT, GAZ_CANON_SOFT, GAZ_TOK_ASP_SOFT, GAZ_PREF_ASP_SOFT = set(), set(), {}, {}, {}

# alias kept for legacy code path expecting dict of aspect->values
GAZ_ORIG = GAZ_ORIG_HARD
GAZ_CANON = GAZ_CANON_HARD
def extract_rare_aspects(title, category):
    """
    Extract rare aspects that are currently missing from predictions.
    Returns list of (aspect, value) tuples.
    """
    rare_predictions = []
    title_lower = title.lower()
    
    # Länge (Length) - look for patterns like "XXXmm", "XXX mm", "Länge: XXX"
    length_patterns = [
        r'(\d+(?:[,\.]\d+)?)\s*mm\s*länge',
        r'länge[\s:]+(\d+(?:[,\.]\d+)?)\s*mm',
        r'(\d{2,4})\s*mm\s*lang',
        r'l[äa]nge[\s:]*(\d+(?:[,\.]\d+)?)',
        r'(\d+(?:[,\.]\d+)?)\s*x\s*\d+\s*mm',  # First dimension often length
    ]
    for pattern in length_patterns:
        match = re.search(pattern, title_lower)
        if match:
            value = match.group(1).replace(',', '.')
            rare_predictions.append(("Länge", f"{value}mm"))  # Fixed: was "LÃ¤nge"
            break
    
    # Breite (Width) - look for "breit", "breite", width dimensions
    width_patterns = [
        r'(\d+(?:[,\.]\d+)?)\s*mm\s*breit',
        r'breite[\s:]+(\d+(?:[,\.]\d+)?)\s*mm',
        r'(\d+(?:[,\.]\d+)?)\s*mm\s*breite',
        r'\d+\s*x\s*(\d+(?:[,\.]\d+)?)\s*mm',  # Second dimension often width
        r'breite[\s:]*(\d+(?:[,\.]\d+)?)',
    ]
    for pattern in width_patterns:
        match = re.search(pattern, title_lower)
        if match:
            value = match.group(1).replace(',', '.')
            rare_predictions.append(("Breite", f"{value}mm"))
            break
    
    # Farbe (Color) - German color terms
    color_terms = {
        'schwarz': 'Schwarz', 'weiß': 'Weiß', 'weiss': 'Weiß',
        'rot': 'Rot', 'blau': 'Blau', 'grün': 'Grün', 'gruen': 'Grün',
        'gelb': 'Gelb', 'grau': 'Grau', 'silber': 'Silber',
        'gold': 'Gold', 'chrom': 'Chrom', 'carbon': 'Carbon',
        'transparent': 'Transparent', 'klar': 'Klar'
    }
    for term, value in color_terms.items():
        if term in title_lower:
            # Check it's not part of another word
            if re.search(r'\b' + term + r'\b', title_lower):
                rare_predictions.append(("Farbe", value))
                break
    
    # Zähnezahl (Number of teeth) - for timing belts/chains
    if category == 2:  # Engine timing category
        teeth_patterns = [
            r'(\d+)\s*zähne',
            r'(\d+)\s*zaehne',
            r'(\d+)\s*z\b',  # Common abbreviation
            r'z[\s:]*(\d+)\b',
            r'(\d+)\s*teeth',  # Sometimes in English
        ]
        for pattern in teeth_patterns:
            match = re.search(pattern, title_lower)
            if match:
                value = match.group(1)
                if 10 <= int(value) <= 300:  # Reasonable range for teeth
                    rare_predictions.append(("Zähnezahl", value))  # Fixed: was inconsistent
                    break
    
    # Technologie (Technology) - specific tech terms
    tech_terms = {
        'abs': 'ABS', 'esp': 'ESP', 'asr': 'ASR',
        'tdi': 'TDI', 'tfsi': 'TFSI', 'tsi': 'TSI',
        'cdi': 'CDI', 'hdi': 'HDI', 'dci': 'DCI',
        'quattro': 'Quattro', '4matic': '4MATIC',
        'hybrid': 'Hybrid', 'electric': 'Electric',
        'turbo': 'Turbo', 'kompressor': 'Kompressor'
    }
    for term, value in tech_terms.items():
        if re.search(r'\b' + term + r'\b', title_lower):
            rare_predictions.append(("Technologie", value))
            break
    
    # Herstellungsland_Und_-Region (Manufacturing country/region)
    country_terms = {
        'deutschland': 'Deutschland', 'germany': 'Deutschland',
        'italien': 'Italien', 'italy': 'Italien',
        'frankreich': 'Frankreich', 'france': 'Frankreich',
        'spanien': 'Spanien', 'spain': 'Spanien',
        'japan': 'Japan', 'china': 'China',
        'usa': 'USA', 'uk': 'UK', 'england': 'England',
        'österreich': 'Österreich', 'austria': 'Österreich',
        'schweiz': 'Schweiz', 'poland': 'Polen', 'polen': 'Polen',
        'made in germany': 'Deutschland', 'made in italy': 'Italien'
    }
    for term, value in country_terms.items():
        if term in title_lower:
            rare_predictions.append(("Herstellungsland_Und_-Region", value))
            break
    
    return rare_predictions
# Validators
OE_PATTERN_MIXED = re.compile(r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9\-\.\/]{5,}$')
OE_PATTERN_NUM   = re.compile(r'^\d{5,}$')
YEAR_PATTERNS = [
    re.compile(r'^(19|20)\d{2}$'),
    re.compile(r'^\d{2}-\d{2}$'), re.compile(r'^\d{4}-\d{2}$'),
    re.compile(r'^\d{2}-\d{4}$'), re.compile(r'^\d{4}-\d{4}$')
]
VISCO_PATTERN = re.compile(r'^\d{1,2}W-?\d{1,2}$', re.IGNORECASE)
EINBAU_WHITELIST = {
    "vorne","hinten","links","rechts","vorderachse","hinterachse","va","ha",
    "fahrerseite","beifahrerseite","vorn","hint"
}

def is_valid_by_rules(asp: str, val: str) -> bool:
    v = val.strip()
    if asp in {"Herstellernummer","Oe/Oem_Referenznummer(N)"}:
        if VOLT_RE.match(v):  # avoid 12V/24V etc misfired as PN
            return False
        # tighten purely numeric PNs
        if v.isdigit():
            if not (PN_NUM_LEN_MIN <= len(v) <= PN_NUM_LEN_MAX):
                return False
        return bool(OE_PATTERN_NUM.match(v) or OE_PATTERN_MIXED.match(v))
    if asp == "Einbauposition":
        v0 = canonicalize_for_match(v).replace("-", "")
        return v0 in {w.replace("-", "") for w in EINBAU_WHITELIST}
    if asp == "Bremsscheiben-Aussendurchmesser":
        m = re.search(r'(\d{2,4})\s*mm', v.lower()) or re.search(r'^\s*(\d{2,4})\s*$', v.lower())
        return bool(m) and (ROTOR_MIN <= int(m.group(1)) <= ROTOR_MAX)
    if asp == "Kompatibles_Fahrzeug_Jahr":
        return any(p.match(v) for p in YEAR_PATTERNS)
    if asp in {"Zähnezahl","Anzahl_Der_Einheiten","Menge"}:
        return bool(re.fullmatch(r'\d+', v))
    return True
def enforce_aspect_category_constraints(predictions_list):
    if not predictions_list:
        return predictions_list
    
    valid_predictions = []
    corrections = 0
    removals = 0
    
    for pred in predictions_list:
        aspect = pred.get("aspect")
        category = pred.get("cat")
        
        # Check if aspect is in our mapping
        if aspect in ASPECT_TO_CATS:
            allowed_cats = ASPECT_TO_CATS[aspect]
            
            if category not in allowed_cats:
                # Wrong category!
                if len(allowed_cats) == 1:
                    # Only one valid category - fix it
                    pred["cat"] = allowed_cats[0]
                    corrections += 1
                    if corrections <= 5:  # Only print first few to avoid spam
                        print(f"  [ASPECT-FIX] '{aspect}' Cat{category} -> Cat{allowed_cats[0]}")
                else:
                    pred["cat"] = allowed_cats[0]
                    corrections += 1
                    if corrections <= 5:
                        print(f"  [ASPECT-FIX] '{aspect}' Cat{category} -> Cat{allowed_cats[0]}")
            
            valid_predictions.append(pred)
        else:
            # Unknown aspect - remove it
            removals += 1
            if removals <= 5:  # Only print first few
                print(f"  [ASPECT-REMOVE] Unknown aspect '{aspect}' in Cat{category}")
    
    if corrections > 0 or removals > 0:
        print(f"[ASPECT-VALIDATION] Fixed {corrections} wrong categories, removed {removals} unknown aspects")
    
    return valid_predictions

def _pa_canon(text: str) -> str:
    return canonicalize_for_match(text)

def _produktart_has_head(text: str, cat: int) -> bool:
    c = _pa_canon(text)
    return any(h in c for h in PA_HEADS.get(int(cat), set()))

def _produktart_is_generic_only(text: str) -> bool:
    toks = set(_pa_canon(text).split())
    if not toks:
        return True
    # only generic tokens, or one-token very generic values
    if toks <= (PA_GENERIC_SOLO | PA_GENERIC_NOISE):
        return True
    if len(toks) == 1 and next(iter(toks)) in PA_GENERIC_SOLO:
        return True
    return False

def _produktart_cross_banned(text: str, cat: int) -> bool:
    c = _pa_canon(text)
    return any(re.search(rf"\b{re.escape(w)}\b", c) for w in PA_CROSS_BAN.get(int(cat), set()))

def is_valid_produktart_value(text: str, cat: int) -> bool:
    """Ultra-conservative Produktart validator: precision >> recall."""
    c = _pa_canon(text)
    if len(c) < 4:
        return False
    if _produktart_is_generic_only(text):
        return False
    if _produktart_cross_banned(text, cat):
        return False

    # must have a domain head OR be known (seen in train or in gazetteer)
    has_head   = _produktart_has_head(text, cat)
    in_gaz     = c in GAZ_CANON.get("Produktart", set())
    seen_train = (TRAIN_VALUE_FREQ["Produktart"][c] > 0)
    if not (has_head or in_gaz or seen_train):
        return False
    return True

def _brand_token_positions(title_tokens):
    """
    Return indices of tokens that look like a vehicle make (Kompatible_Fahrzeug_Marke).
    Uses token-level gazetteer hits; fast and precise enough for proximity checks.
    """
    makes_tok = GAZ_TOK_ASP_HARD.get("Kompatible_Fahrzeug_Marke", set())
    makes_pref = GAZ_PREF_ASP_HARD.get("Kompatible_Fahrzeug_Marke", set())

    pos = []
    for i, tk in enumerate(title_tokens):
        cw = canonicalize_for_match(tk)
        if cw in makes_tok or cw in makes_pref:
            pos.append(i)
    return pos


def _is_pnish(s: str) -> bool:
    """OE/PN-like shapes frequently confuse KFM; keep them out of vehicle 'Modell'."""
    t = s.strip()
    return bool(OE_PATTERN_NUM.match(t) or OE_PATTERN_MIXED.match(t))

def fbeta(prec, rec, beta=0.2):
    if (prec+rec)==0: return 0.0
    b2 = beta*beta
    return (1+b2)*prec*rec/(b2*prec+rec)

def compute_entity_f02(true_ents, pred_ents):
    t_count, p_count = Counter(true_ents), Counter(pred_ents)
    tp_total = 0
    for k, c in p_count.items():
        if k in t_count:
            tp_total += min(c, t_count[k])
    prec = tp_total/(sum(p_count.values())+1e-12)
    rec  = tp_total/(sum(t_count.values())+1e-12)
    return prec, rec, fbeta(prec, rec, 0.2)

# ---------- Heuristics ----------
MM_RE = re.compile(r'(\d{2,4})\s*mm', re.IGNORECASE)
def find_diameters(title_tokens):
    text = " ".join(title_tokens)
    vals = []
    for m in MM_RE.finditer(text):
        try:
            n = int(m.group(1))
        except Exception:
            continue
        if ROTOR_MIN <= n <= ROTOR_MAX:
            vals.append(f"{n}mm")
    seen=set(); out=[]
    for v in vals:
        if v not in seen:
            seen.add(v); out.append(v)
    return out[:3]

def heuristic_pairs(cat, tokens, present_set_canons):
    extra = []
    if cat == 2:
        for token in tokens:
            if re.fullmatch(r'\d{1,2}W-?\d{1,2}', token, flags=re.IGNORECASE):
                a = "SAE_Viskosität"; v = token; key = (a, canonicalize_for_match(v))
                if key not in present_set_canons:
                    extra.append((a, v))
    if cat == 1 and "Bremsscheiben-Aussendurchmesser" in ALLOWED_ASP[1]:
        for dv in find_diameters(tokens):
            a = "Bremsscheiben-Aussendurchmesser"; key = (a, canonicalize_for_match(dv))
            if key not in present_set_canons:
                extra.append((a, dv))
    return extra
def _has_modelish_token(text: str, *, consider_kfm_gaz=True) -> bool:
    cw = canonicalize_for_match(text)
    toks = cw.split()
    ld = any(re.search(r"[a-z][0-9]|[0-9][a-z]", t) for t in toks)
    pat = any(r.match(t) for r in KFM_ALLOW_RE for t in toks)
    if consider_kfm_gaz:
        in_gaz_phrase = cw in GAZ_CANON.get("Kompatibles_Fahrzeug_Modell", set())
        tok_hit = any(t in GAZ_TOK_ASP_HARD.get("Kompatibles_Fahrzeug_Modell", set()) for t in toks)
        if in_gaz_phrase or tok_hit:
            return True
    return bool(ld or pat)


def _kfm_veto(text: str, start_pos: float, title_tokens, *,
              allow_if_brand_and_gaz=False, in_gaz=False, near_brand=False) -> bool:
    t = canonicalize_for_match(text)
    toks = t.split()

    # Early hard veto: single-character "models" like 'V', 'B', 'Y' are never valid KFModell
    if len(toks) == 1 and len(t) == 1:
        return True

    if any(tok in KFM_BAN_TOK for tok in toks):
        return True

    digit_ratio = (sum(ch.isdigit() for ch in t) / max(1, len(t)))
    hard_bad = _is_pnish(text) or digit_ratio >= KFM_MAX_DIGIT_RATIO or len(t) >= KFM_MAX_LEN

    # allow known gaz models near brand even if they fail "modelish"
    if allow_if_brand_and_gaz and in_gaz and near_brand and not hard_bad:
        return False

    if KFM_REQUIRE_MODELISH and not _has_modelish_token(text):
        return True

    # require a nearby brand when we don't have gaz confirmation
    if not near_brand and not in_gaz:
        return True

    return False


# ========================== Data Load =============================
train_df = pd.read_csv(TRAIN_TSV, sep="\t", quoting=csv.QUOTE_NONE, engine="python",
                       keep_default_na=False)
rec_groups = train_df.groupby(train_df.columns[0])

records = []
for rid, g in rec_groups:
    cat = int(g.iloc[0,1])
    toks = g.iloc[:,3].astype(str).tolist()
    tags = g.iloc[:,4].astype(str).tolist()
    bio = []
    prev = None
    for t in tags:
        if t == "O":
            bio.append("O"); prev=None
        elif t == "":
            bio.append("I-"+prev if prev else "O")
        else:
            bio.append("B-"+t); prev=t
    records.append({"rid": int(rid), "cat":cat, "tokens":toks, "bio":bio})

TRAIN_VALUE_FREQ = defaultdict(lambda: Counter())
for ex in records:
    toks, bio = ex["tokens"], ex["bio"]
    i, n = 0, min(len(toks), len(bio))
    while i < n:
        lab = bio[i]
        if lab == "O": i += 1; continue
        a = lab[2:]
        j = i + 1
        while j < n and bio[j] == ("I-" + a): j += 1
        val = " ".join(toks[i:j])
        TRAIN_VALUE_FREQ[a][canonicalize_for_match(val)] += 1
        i = j

# =================== Label Space / Tokenizer ======================
LABELS = ["O"] + [f"B-{a}" for a in ALL_ASP] + [f"I-{a}" for a in ALL_ASP]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for i,l in enumerate(LABELS)}
id2aspect = {i: ("O" if l=="O" else l.split("-",1)[1]) for i,l in id2label.items()}

def catmask(cat:int):
    allow = {label2id["O"]}
    for a in ALLOWED_ASP[cat]:
        allow.add(label2id["B-"+a]); allow.add(label2id["I-"+a])
    mask = np.zeros(len(LABELS), dtype=np.int64); mask[list(allow)] = 1
    return torch.tensor(mask)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# ---------- Per-token extras: char-ids & per-aspect gaz features ----------
ALL_ASP_IDX = {a:i for i,a in enumerate(ALL_ASP)}
GAZ_DIM_PER_ASP = 2  # [token_hit, prefix_hit]
EXTRA_GLOBAL_GAZ = 2 # [any_hard_hit, any_soft_hit]
GAZ_TOTAL_DIM = len(ALL_ASP)*GAZ_DIM_PER_ASP + EXTRA_GLOBAL_GAZ

def word_to_char_ids(word, maxlen=CHAR_MAXLEN):
    s = unicodedata.normalize("NFKD", word).encode("ascii","ignore").decode("ascii")
    s = s[:maxlen].lower()
    # map to 96-char set [a-z0-9 -_./+%()[]]
    vocab = "abcdefghijklmnopqrstuvwxyz0123456789-._/+%()[] "
    table = {ch:i+2 for i,ch in enumerate(vocab)}  # 0=pad,1=unk
    out = []
    for ch in s:
        out.append(table.get(ch, 1))
    while len(out) < maxlen:
        out.append(0)
    return out[:maxlen]

def build_aspect_gaz_hits_for_word(cw):
    # per-aspect [token_hit, prefix_hit]
    feat = np.zeros((len(ALL_ASP), GAZ_DIM_PER_ASP), dtype=np.float32)
    for a, idx in ALL_ASP_IDX.items():
        hit_tok = 1.0 if (cw in GAZ_TOK_ASP_HARD.get(a, set())) else 0.0
        hit_pre = 1.0 if (cw in GAZ_PREF_ASP_HARD.get(a, set())) else 0.0
        feat[idx, 0] = hit_tok
        feat[idx, 1] = hit_pre
    return feat.reshape(-1)  # (len(ALL_ASP)*2,)

def build_token_extras(word_ids, words):
    """
    Returns:
      char_ids: (seq_len, CHAR_MAXLEN) int64
      gaz_aspect: (seq_len, GAZ_TOTAL_DIM) float32
    """
    char_ids = []
    gaz_feats = []
    prev_w = None
    cache = {}
    for wi in word_ids:
        if wi is None:
            char_ids.append([0]*CHAR_MAXLEN)
            gaz_feats.append(np.zeros(GAZ_TOTAL_DIM, dtype=np.float32))
        else:
            if wi not in cache:
                w = words[wi]
                cw = canonicalize_for_match(w)
                # per-aspect 2-d hits
                asp_vec = build_aspect_gaz_hits_for_word(cw) if USE_ASP_GAZ else np.zeros(len(ALL_ASP)*GAZ_DIM_PER_ASP, dtype=np.float32)
                # global any hits (hard/soft)
                any_hard = 1.0 if (cw in GAZ_VAL_HARD or cw in GAZ_TOK_HARD) else 0.0
                any_soft = 1.0 if (cw in GAZ_VAL_SOFT or cw in GAZ_TOK_SOFT) else 0.0
                gaz = np.concatenate([asp_vec, np.asarray([any_hard, any_soft], dtype=np.float32)], axis=0)
                cache[wi] = (word_to_char_ids(w, CHAR_MAXLEN), gaz)
            c, g = cache[wi]
            char_ids.append(c)
            gaz_feats.append(g)
    return np.asarray(char_ids, dtype=np.int64), np.asarray(gaz_feats, dtype=np.float32)

def encode_example(tokens, bio, max_length=256):
    enc = tokenizer(tokens, is_split_into_words=True, add_special_tokens=True,
                    return_offsets_mapping=False, truncation=True, max_length=max_length)
    word_ids = enc.word_ids()
    labels = []
    prev_w = None
    for wi in word_ids:
        if wi is None:
            labels.append(-100)
        else:
            if wi != prev_w:
                labels.append(label2id[bio[wi]])
            else:
                l = bio[wi]
                labels.append(label2id["O"] if l=="O" else label2id["I-"+l.split("-",1)[1]])
            prev_w = wi
    char_ids, gaz_aspect = build_token_extras(word_ids, tokens)
    return enc["input_ids"], enc["attention_mask"], labels, char_ids, gaz_aspect

# ============================ Model ===============================
class CharCNN(nn.Module):
    def __init__(self, vocab_size=100, emb_dim=24, maxlen=CHAR_MAXLEN, out_channels=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv2 = nn.Conv1d(emb_dim, out_channels, kernel_size=2, padding=0)
        self.conv3 = nn.Conv1d(emb_dim, out_channels, kernel_size=3, padding=0)
        self.conv4 = nn.Conv1d(emb_dim, out_channels, kernel_size=4, padding=0)
        self.maxlen = maxlen
        self.out_dim = out_channels*3

    def forward(self, char_ids):  # (B, L, C)
        B, L, C = char_ids.shape
        x = self.emb(char_ids.view(B*L, C))             # (B*L, C, emb)
        x = x.transpose(1, 2)                           # (B*L, emb, C)
        f2 = torch.relu(self.conv2(x)).amax(dim=-1)     # (B*L, out)
        f3 = torch.relu(self.conv3(x)).amax(dim=-1)
        f4 = torch.relu(self.conv4(x)).amax(dim=-1)
        feat = torch.cat([f2, f3, f4], dim=-1)          # (B*L, out*3)
        return feat.view(B, L, -1)                      # (B, L, out*3)

class AspectModelProPlus(nn.Module):
    def __init__(self, num_labels, hidden_name=MODEL_NAME):
        super().__init__()
        self.enc = AutoModel.from_pretrained(hidden_name)
        H = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)

        self.use_char = USE_CHARCNN
        if self.use_char:
            self.charcnn = CharCNN(vocab_size=100, emb_dim=24, maxlen=CHAR_MAXLEN, out_channels=32)
            self.char_proj = nn.Linear(self.charcnn.out_dim, H)
            self.char_gate = nn.Parameter(torch.tensor(0.0))  # start near off; learned

        # Per-aspect gazetteer features -> direct logit bias
        self.use_asp_gaz = USE_ASP_GAZ
        self.gaz_fc = nn.Linear(GAZ_TOTAL_DIM, num_labels, bias=False)

        self.classifier = nn.Linear(H, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attn_mask, char_ids=None, gaz_aspect=None,
                labels=None, label_mask=None, amp_dtype=None):
        with autocast(device_type="cuda", enabled=(amp_dtype is not None and torch.cuda.is_available()), dtype=amp_dtype):
            h = self.enc(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state  # (B,L,H)

            if self.use_char and char_ids is not None:
                cf = self.charcnn(char_ids)                           # (B,L,Cc)
                cproj = self.char_proj(cf)                            # (B,L,H)
                g = torch.sigmoid(self.char_gate)
                h = h + g * cproj                                     # gated fusion

            logits = self.classifier(self.drop(h))                    # (B,L,K)

        if self.use_asp_gaz and gaz_aspect is not None:
            logits = logits + self.gaz_fc(gaz_aspect)                 # add gaz logit bias

        if label_mask is not None:
            mask = label_mask.unsqueeze(1).expand(-1, logits.size(1), -1)
            logits = logits.masked_fill(mask==0, -1e4)

        logits32 = logits.float()
        if labels is not None:
            labels2 = labels.clone()
            labels2[labels2 == -100] = label2id["O"]
            valid = attn_mask.bool()
            if valid.dim() == 2:
                valid[:, 0] = True
            ll = self.crf(logits32, labels2, mask=valid, reduction='mean')
            return -ll
        else:
            seq = self.crf.decode(logits32, mask=attn_mask.bool())
            return logits32, seq

# ===================== Decode & helpers ===========================
def decode_entities(tokens, labels_str):
    n = min(len(tokens), len(labels_str))
    out = []; i = 0
    while i < n:
        lab = labels_str[i]
        if lab == "O" or lab is None:
            i += 1; continue
        a = lab[2:] if lab.startswith(("B-","I-")) else None
        if a is None:
            i += 1; continue
        j = i + 1
        while j < n and labels_str[j] == ("I-" + a):
            j += 1
        out.append((a, " ".join(tokens[i:j]))); i = j
    return out

def per_token_conf_from_emissions(logits, labels_ids):
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    confs=[]
    for i,lid in enumerate(labels_ids):
        if lid < 0 or lid >= len(LABELS):
            confs.append(0.0); continue
        a = id2aspect[lid]
        if a=="O":
            confs.append(1.0); continue
        b = label2id.get("B-"+a); ii=label2id.get("I-"+a)
        p = (probs[i,b] if b is not None else 0.0) + (probs[i,ii] if ii is not None else 0.0)
        confs.append(float(p))
    return confs

def spans_with_conf(labels_ids, confs, tokens):
    out=[]; i=0; L=len(labels_ids)
    while i<L:
        lid = labels_ids[i]
        if lid==label2id["O"]:
            i+=1; continue
        a = id2aspect[lid]
        j=i+1; vals=[confs[i]]
        while j<L and id2aspect[labels_ids[j]]==a and LABELS[labels_ids[j]].startswith("I-"):
            vals.append(confs[j]); j+=1
        mn = float(np.min(vals)) if vals else 0.0
        mu = float(np.mean(vals)) if vals else 0.0
        gm = float(np.exp(np.mean(np.log(np.maximum(1e-6, vals))))) if vals else 0.0
        sd = float(np.std(vals)) if vals else 0.0
        out.append((a, " ".join(tokens[i:j]), mn, mu, gm, sd, i))
        i=j
    return out

# ======================= Build encodings ==========================
encoded = []
for ex in records:
    inp, msk, lab, ch, gaz = encode_example(ex["tokens"], ex["bio"])
    encoded.append( (torch.tensor(inp), torch.tensor(msk), torch.tensor(lab),
                     torch.tensor(ch), torch.tensor(gaz), ex["cat"], ex["tokens"]) )

def batchify(batch):
    maxL = max(len(x[0]) for x in batch)
    B = len(batch)
    pad_id = tokenizer.pad_token_id
    ids = torch.full((B,maxL), pad_id, dtype=torch.long)
    msk = torch.zeros((B,maxL), dtype=torch.long)
    labs= torch.full((B,maxL), -100, dtype=torch.long)
    ch  = torch.zeros((B,maxL,CHAR_MAXLEN), dtype=torch.long)
    gaz = torch.zeros((B,maxL,GAZ_TOTAL_DIM), dtype=torch.float)
    cm  = torch.zeros((B,len(LABELS)), dtype=torch.long)
    for i,(ii,mm,ll,cc,gz,cat,_) in enumerate(batch):
        L=len(ii)
        ids[i,:L]=ii; msk[i,:L]=mm; labs[i,:L]=ll
        cL = min(L, cc.shape[0]); gL=min(L, gz.shape[0])
        ch[i,:cL,:] = cc[:cL,:]
        gaz[i,:gL,:]= gz[:gL,:]
        cm[i]=catmask(cat)
    ids = ids.to(device, non_blocking=True)
    msk = msk.to(device, non_blocking=True)
    labs= labs.to(device, non_blocking=True)
    ch  = ch.to(device, non_blocking=True)
    gaz = gaz.to(device, non_blocking=True)
    cm  = cm.to(device, non_blocking=True)
    return ids, msk, labs, ch, gaz, cm

# ========================== Evaluate ==============================
def compute_dynamic_caps(title_tokens, cat):
    base = MAX_PER_TITLE_BASE[cat]
    default_factory = (base.default_factory if isinstance(base, defaultdict) and base.default_factory is not None
                       else (lambda: 3))
    caps = defaultdict(default_factory, dict(base))
    toks = [canonicalize_for_match(t) for t in title_tokens]
    if any(c in " ".join(toks) for c in {"+","&","/","und"}) or any(t in {"vorn","vorne","hinten","hint","va","ha","vorderachse","hinterachse"} for t in toks):
        if "Einbauposition" in ALLOWED_ASP[cat]:
            caps["Einbauposition"] = max(caps["Einbauposition"], 2)
        if "Bremsscheiben-Aussendurchmesser" in ALLOWED_ASP[cat]:
            caps["Bremsscheiben-Aussendurchmesser"] = max(caps["Bremsscheiben-Aussendurchmesser"], 2)
    return caps
def apply_aspect_caps(kept, cat, title_tokens):
    by_asp = defaultdict(list)
    for d in kept:
        by_asp[d["aspect"]].append(d)
    STRICT_LIMITS = {
        "Bremsscheiben-Aussendurchmesser": 1,  # Only 1 diameter per record
        "Anzahl_Der_Einheiten": 1,  # Only 1 quantity
        "Einbauposition": 1,  # Only 1 position
        "Bremsscheibenart": 1,  # Only 1 type
    }
    
    for aspect, limit in STRICT_LIMITS.items():
        if aspect in by_asp and len(by_asp[aspect]) > limit:
            # Keep only the highest confidence one
            by_asp[aspect] = sorted(by_asp[aspect], 
                                   key=lambda x: x.get("conf_min", 0.0), 
                                   reverse=True)[:limit]
    
    final = []
    caps = compute_dynamic_caps(title_tokens, cat)
    
    for a, lst in by_asp.items():
        # Sort by confidence and support
        lst.sort(key=lambda x: (x.get("conf_min", 0.0), x.get("support", 0), len(x.get("text", ""))), reverse=True)
        cap = caps[a]
        for d in lst[:cap]:
            final.append(d)
    
    return final

def _kept_after_caps(detailed_list, cat, title_tokens):
    by_asp = defaultdict(list)
    for d in detailed_list:
        by_asp[d["aspect"]].append(d)
    final = []
    caps = compute_dynamic_caps(title_tokens, cat)
    for a, lst in by_asp.items():
        lst.sort(key=lambda x: (x.get("conf_min",0.0), x.get("support",0), len(x["text"])), reverse=True)
        cap = caps[a]
        for d in lst[:cap]:
            final.append(d)
    return final

def conflict_resolver(cat, title_tokens, selected):
    toks_lc = [t.lower() for t in title_tokens]
    out = []
    seen = set((d["aspect"], d["canon"]) for d in selected)
    for d in selected:
        keep = True
        if d["aspect"] == "Hersteller":
            v = canonicalize_for_match(d["text"])
            near_ctx = False
            for i, tk in enumerate(toks_lc):
                if tk in FUER_SYNONYMS:
                    window = toks_lc[max(0,i-8): i+9]
                    if any(v in canonicalize_for_match(" ".join(window[j:j+len(d['text'].split())]))
                           for j in range(0, max(1,len(window)))):
                        near_ctx = True; break
            if near_ctx and ("Kompatible_Fahrzeug_Marke", v) in seen and \
               not (("Hersteller" in GAZ_CANON) and (v in GAZ_CANON["Hersteller"])):
                keep = False
        if keep and d["aspect"] == "Produktart":
            # Drop generic Produktart if a more specific one exists or if it's just an inclusion item
            if _produktart_is_generic_only(d["text"]):
                keep = False
            else:
                # If the exact same text is also predicted as "Im_Lieferumfang_Enthalten",
                # keep Produktart only if it's known or has a domain head.
                same_as_inclusion = any(
                    (d2 is not d) and d2["aspect"] == "Im_Lieferumfang_Enthalten" and d2["canon"] == d["canon"]
                    for d2 in selected
                )
                if same_as_inclusion:
                    c = d["canon"]
                    known = (TRAIN_VALUE_FREQ["Produktart"][c] > 0) or (c in GAZ_CANON.get("Produktart", set()))
                    if not (known or _produktart_has_head(d["text"], cat)):
                        keep = False
        if d["aspect"] in {"Produktart","Kompatibles_Fahrzeug_Modell"}:
            for d2 in selected:
                if d2 is d: continue
                if d2["aspect"] == d["aspect"]:
                    if d["text"] in d2["text"] and (d2["canon"] in GAZ_CANON.get(d2["aspect"], set())):
                        keep = False; break
        if keep: out.append(d)
    return out

def evaluate_cv(model, val_indices, strict=True):
    model.eval()
    true_ents, pred_ents = [], []
    with torch.no_grad():
        for i in val_indices:
            ex = records[i]; tokens = ex["tokens"]
            enc = tokenizer(tokens, is_split_into_words=True, add_special_tokens=True,
                            return_offsets_mapping=False, truncation=True, max_length=256)
            word_ids = enc.word_ids()
            ids = torch.tensor(enc["input_ids"], dtype=torch.long).unsqueeze(0).to(device, non_blocking=True)
            msk = torch.tensor(enc["attention_mask"], dtype=torch.long).unsqueeze(0).to(device, non_blocking=True)
            # build token extras aligned to encoded tokens
            ch_np, gz_np = build_token_extras(word_ids, tokens)
            ch = torch.tensor(ch_np, dtype=torch.long).unsqueeze(0).to(device, non_blocking=True)
            gaz= torch.tensor(gz_np, dtype=torch.float).unsqueeze(0).to(device, non_blocking=True)
            cm  = catmask(ex["cat"]).unsqueeze(0).to(device, non_blocking=True)

            logits, seq = model(ids, msk, char_ids=ch, gaz_aspect=gaz, labels=None, label_mask=cm, amp_dtype=None)
            seq = seq[0]
            word_labels, prev = [], None
            for idx_w, wid in enumerate(word_ids):
                if wid is None: continue
                if wid != prev:
                    lab_id = seq[idx_w] if idx_w < len(seq) else label2id["O"]
                    word_labels.append(id2label[lab_id]); prev = wid
                else:
                    prev = wid

            true_ents.extend(decode_entities(tokens, ex["bio"]))

            raw_pairs = post_filter_entities(ex["cat"], decode_entities(tokens, word_labels), strict=strict)

            present_set_canons = {(a, canonicalize_for_match(v)) for (a,v) in raw_pairs}
            for a, v in heuristic_pairs(ex["cat"], tokens, present_set_canons):
                raw_pairs.append((a, v))

            detailed = [{
                "aspect": a, "text": v, "canon": canonicalize_for_match(v),
                "conf_min": 0.99, "support": 5, "cat": ex["cat"], "start": 0.0, "title_tokens": tokens
            } for (a, v) in raw_pairs]

            kept = _kept_after_caps(detailed, ex["cat"], tokens)
            kept = conflict_resolver(ex["cat"], tokens, kept)

            pred_ents.extend([(d["aspect"], d["text"]) for d in kept])
    return compute_entity_f02(true_ents, pred_ents)

# ========================= Post-filtering =========================
def post_filter_entities(cat: int, ents, strict=True):
    out, seen = [], set()
    UNIT_WHITELIST = {"mm","l","stück","stuck","zoll","cm"}
    _SEP_RE = re.compile(r"[ \-_/\.]")

    for a, v_out in ents:
        if not (a in ALLOWED_ASP.get(int(cat), set())): continue
        v_out = canonicalize_output(v_out)
        if not v_out: continue
        v_can = canonicalize_for_match(v_out)
        keep = True
        if strict:
            if a in GAZ_CANON and v_can in GAZ_CANON[a]:
                keep = is_valid_by_rules(a, v_out)
            else:
                if a == "Einbauposition":
                    v0 = v_can.replace("-", "")
                    keep = v0 in {w.replace("-", "") for w in EINBAU_WHITELIST}
                elif a == "Bremsscheiben-Aussendurchmesser":
                    m = re.search(r'(\d{2,4})\s*mm', v_can) or re.search(r'^\s*(\d{2,4})\s*$', v_can)
                    keep = bool(m) and (ROTOR_MIN <= int(m.group(1)) <= ROTOR_MAX)
                elif a == "Produktart":
                    keep = is_valid_produktart_value(v_out, cat)
                elif a == "Kompatibles_Fahrzeug_Jahr":
                    keep = any(p.match(v_out) for p in YEAR_PATTERNS)
                elif a in {"Zähnezahl","Anzahl_Der_Einheiten","Menge"}:
                    keep = bool(re.fullmatch(r"\d+", v_can))
                elif a in {"Herstellernummer","Oe/Oem_Referenznummer(N)"}:
                    keep = bool(OE_PATTERN_NUM.match(v_out) or OE_PATTERN_MIXED.match(v_out))
                elif a == "Maßeinheit":
                    keep = v_can in UNIT_WHITELIST
                else:
                    keep = is_valid_by_rules(a, v_out)
        else:
            if a in {"Herstellernummer","Oe/Oem_Referenznummer(N)"}:
                keep = (len(_SEP_RE.sub("", v_out)) >= 5)
            else:
                keep = True
        if not keep: continue
        key = (a, v_can)
        if key in seen: continue
        seen.add(key)
        out.append((a, v_out))
    return out

# ===================== Train (5-fold) =============================
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_models, fold_cv_scores, fold_splits = [], [], []

print("== Training / Loading folds ==")
for fold_id, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(encoded))), start=1):
    ckpt_path = os.path.join(CKPT_DIR, f"fold{fold_id}_best.pt")
    model = AspectModelProPlus(num_labels=len(LABELS)).to(device)
    if USE_COMPILE: model = torch.compile(model)

    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"[Fold {fold_id}] Loaded checkpoint.")
    else:
        print(f"[Fold {fold_id}] Training...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        train_batches = [encoded[i] for i in train_idx]
        num_batches = max(1, math.ceil(len(train_batches) / BATCH_SIZE))
        total_steps = num_batches * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=max(1,total_steps//10), num_training_steps=total_steps
        )
        scaler = GradScaler() if (torch.cuda.is_available() and AMP_DTYPE==torch.float16) else None

        best_f, bad = -1.0, 0
        for ep in range(1, EPOCHS+1):
            model.train(); random.shuffle(train_batches)
            running = 0.0; step_count = 0
            for s in range(0, len(train_batches), BATCH_SIZE):
                batch = train_batches[s:s+BATCH_SIZE]
                ids, msk, labs, ch, gaz, cm = batchify(batch)
                optimizer.zero_grad(set_to_none=True)
                loss = model(ids, msk, char_ids=ch, gaz_aspect=gaz, labels=labs, label_mask=cm, amp_dtype=AMP_DTYPE)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                running += float(loss.detach().cpu()); step_count += 1

            prec, rec, f02 = evaluate_cv(model, val_idx, strict=True)
            print(f"[Fold {fold_id}] Ep {ep} | loss={running/max(1,step_count):.4f} | CV(strict) P={prec:.4f} R={rec:.4f} F0.2={f02:.4f}")
            if f02 > best_f + 1e-4:
                best_f, bad = f02, 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                bad += 1
                if bad >= EARLY_STOP_PATIENCE:
                    print(f"[Fold {fold_id}] Early stop at epoch {ep}."); break

        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd, strict=False)

    prec, rec, f02 = evaluate_cv(model, val_idx, strict=True)
    print(f"[Fold {fold_id}] CV(strict) P={prec:.4f} R={rec:.4f} F0.2={f02:.4f}")
    fold_models.append(model); fold_cv_scores.append(f02)
    fold_splits.append((train_idx, val_idx))

# ================= Strict inference helpers =======================
def _merge_contained_groups(group):
    by_asp = defaultdict(list)
    for k in list(group.keys()):
        by_asp[k[0]].append(k)
    for asp, keys in by_asp.items():
        keys = [k for k in keys if k in group]
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                k1 = keys[i]; k2 = keys[j]
                if k1 not in group or k2 not in group: continue
                t1 = max(group[k1]["texts"], key=len)
                t2 = max(group[k2]["texts"], key=len)
                if t1 in t2 or t2 in t1:
                    big, small = (k1, k2) if len(t1) >= len(t2) else (k2, k1)
                    group[big]["texts"] |= group[small]["texts"]
                    group[big]["folds"] |= group[small]["folds"]
                    group[big]["confs_min"] += group[small]["confs_min"]
                    group[big]["confs_mean"] += group[small]["confs_mean"]
                    group[big]["confs_gmean"] += group[small]["confs_gmean"]
                    group[big]["confs_std"] += group[small]["confs_std"]
                    group[big]["starts"] += group[small]["starts"]
                    del group[small]
    return group

def decode_one_fold(model, title_tokens, cat_int):
    enc = tokenizer(title_tokens, is_split_into_words=True, add_special_tokens=True,
                    return_offsets_mapping=False, truncation=True, max_length=256)
    word_ids = enc.word_ids()
    ids = torch.tensor(enc["input_ids"], dtype=torch.long).unsqueeze(0).to(device, non_blocking=True)
    msk = torch.tensor(enc["attention_mask"], dtype=torch.long).unsqueeze(0).to(device, non_blocking=True)
    ch_np, gz_np = build_token_extras(word_ids, title_tokens)
    ch = torch.tensor(ch_np, dtype=torch.long).unsqueeze(0).to(device, non_blocking=True)
    gaz= torch.tensor(gz_np, dtype=torch.float).unsqueeze(0).to(device, non_blocking=True)
    cm  = catmask(int(cat_int)).unsqueeze(0).to(device, non_blocking=True)
    with torch.no_grad():
        logits, seq = model(ids, msk, char_ids=ch, gaz_aspect=gaz, labels=None, label_mask=cm, amp_dtype=None)
    seq = seq[0]
    word_lab_ids, word_logits, prev = [], [], None
    for i, wid in enumerate(word_ids):
        if wid is None: continue
        if wid != prev:
            word_lab_ids.append(seq[i] if i < len(seq) else label2id["O"])
            word_logits.append(logits[0,i].detach().cpu())
        prev = wid
    if len(word_logits)==0:
        return []
    word_logits = torch.stack(word_logits, dim=0)
    confs = per_token_conf_from_emissions(word_logits, word_lab_ids)
    spans = spans_with_conf(word_lab_ids, confs, title_tokens)
    relaxed = post_filter_entities(int(cat_int), [(a, t) for (a,t,_,__,___,____,_____) in spans], strict=False)
    conf_map_full, start_map = {}, {}
    for (a,t,mn,mu,gm,sd,st) in spans:
        key = (a, canonicalize_for_match(t))
        conf_map_full.setdefault(key, []).append((mn,mu,gm,sd))
        start_map.setdefault(key, []).append(st)
    out = []
    for a, v in relaxed:
        key = (a, canonicalize_for_match(v))
        stats = conf_map_full.get(key, [(0,0,0,0)])
        mns = [x[0] for x in stats]; mus = [x[1] for x in stats]; gms=[x[2] for x in stats]; sds=[x[3] for x in stats]
        starts = start_map.get(key, [0])
        out.append({"aspect":a, "text":v, "min_conf":max(mns), "mean_conf":max(mus),
                    "gmean_conf":max(gms), "std_conf":min(sds), "start":float(np.min(starts))})
    return out

def _build_groups_for_models(models, title_tokens, cat_int):
    per_fold = [decode_one_fold(m, title_tokens, cat_int) for m in models]
    group = {}
    for f_id, lst in enumerate(per_fold):
        for c in lst:
            a = c["aspect"]; v_c = canonicalize_for_match(c["text"])
            key = (a, v_c)
            if key not in group:
                group[key] = {"aspect":a, "canon":v_c, "texts":set(), "folds":set(),
                              "confs_min":[], "confs_mean":[], "confs_gmean":[], "confs_std":[],
                              "starts":[]}
            group[key]["texts"].add(c["text"])
            group[key]["folds"].add(f_id)
            group[key]["confs_min"].append(c["min_conf"])
            group[key]["confs_mean"].append(c["mean_conf"])
            group[key]["confs_gmean"].append(c["gmean_conf"])
            group[key]["confs_std"].append(c["std_conf"])
            group[key]["starts"].append(c["start"])
    group = _merge_contained_groups(group)
    return group
def _is_value_conflicted(aspect, v_can, kept):
    # Check for duplicate aspect-value pairs
    for existing in kept:
        if existing["aspect"] == aspect and existing["canon"] == v_can:
            return True
    if aspect in ["Produktart", "Kompatibles_Fahrzeug_Modell"]:
        for existing in kept:
            if existing["aspect"] == aspect:
                if v_can in existing["canon"] or existing["canon"] in v_can:
                    # Keep the longer/more specific one (already selected)
                    if len(existing["canon"]) >= len(v_can):
                        return True

        # Extra safety for KFModell: if this is an ultra-short "model" and there is already
        # at least one other KFModell in this title, drop the short one.
        if aspect == "Kompatibles_Fahrzeug_Modell":
            # Current candidate is extremely short (typically one token like 'v', 'b').
            if len(v_can) <= 2:
                if any(e["aspect"] == "Kompatibles_Fahrzeug_Modell" for e in kept):
                    return True
    
    return False
def strict_select_from_groups_enhanced(groups, cat, title_tokens, n_models, vote_fraction, conf_thresh):
    """
    Enhanced selection with frequency awareness and strict validation.
    """
    kept = []
    
    for (aspect, v_can), info in groups.items():
        # Voting logic
        vote = float(len(info["folds"]))
        vote_frac = vote / float(n_models)
        
        # Get thresholds
        aspect_vote_thresh = vote_fraction.get((cat, aspect), vote_fraction.get(aspect, 0.6))
        aspect_conf_thresh = conf_thresh.get((cat, aspect), conf_thresh.get(aspect, 0.5))
        
        # Check thresholds
        if vote_frac < aspect_vote_thresh:
            continue
        
        # Calculate confidence metrics
        conf_min = np.min(info["confs_min"]) if info["confs_min"] else 0.0
        conf_mean = np.mean(info["confs_mean"]) if info["confs_mean"] else 0.0
        conf_gmean = np.mean(info["confs_gmean"]) if info["confs_gmean"] else 0.0
        conf_std = np.mean(info["confs_std"]) if info["confs_std"] else 0.0
        
        if conf_min < aspect_conf_thresh:
            continue
        
        # PATCH: Add strict validation for problematic aspects
        if aspect == 'Anzahl_Der_Einheiten':
            # Get the original text value (not canonicalized)
            text_val = max(info["texts"], key=lambda s: len(s)) if info["texts"] else v_can
            # Check the original text format
            if not re.match(r'^\d+[xX]?$|^\d+\s*[xX]$|^\d+[Pp]cs$|^\d+-teilig$', text_val):
                continue
                
        # Check for conflicts
        if _is_value_conflicted(aspect, v_can, kept):
            continue
        
        # Get the best text representation
        text = max(info["texts"], key=lambda s: len(s))
        
        # Add to kept list
        kept.append({
            "aspect": aspect,
            "canon": v_can,
            "text": text,
            "conf_mean": conf_mean,
            "conf_min": conf_min,
            "conf_gmean": conf_gmean,
            "conf_std": conf_std,
            "support": vote,
            "start": np.mean(info["starts"]) if info["starts"] else 0.0,
            "cat": cat,
            "title_tokens": title_tokens
        })
    
    # Apply Im_Lieferumfang_Enthalten filtering
    title = ' '.join(title_tokens)
    kept = filter_im_lieferumfang_false_positives(kept, title, cat)
    
    # Add rare aspects - KEEP THIS, IT HELPS!
    kept = integrate_rare_aspects_into_predictions(kept, title, cat)
    
    # Apply dynamic caps
    kept = apply_aspect_caps(kept, cat, title_tokens)
    
    return kept
# ===================== Stage-1 selection ==========================
def strict_select_from_groups(groups, cat, title_tokens, n_models, vote_fraction, conf_thresh):
    kept = []
    
    for (aspect, v_can), info in groups.items():
        # Voting logic
        vote = float(len(info["folds"]))  # Use folds count as vote
        vote_frac = vote / float(n_models)
        
        # Get thresholds
        aspect_vote_thresh = vote_fraction.get((cat, aspect), vote_fraction.get(aspect, 0.6))
        aspect_conf_thresh = conf_thresh.get((cat, aspect), conf_thresh.get(aspect, 0.5))
        
        # Check thresholds
        if vote_frac < aspect_vote_thresh:
            continue
        
        # Calculate confidence metrics
        conf_min = np.min(info["confs_min"]) if info["confs_min"] else 0.0
        conf_mean = np.mean(info["confs_mean"]) if info["confs_mean"] else 0.0
        conf_gmean = np.mean(info["confs_gmean"]) if info["confs_gmean"] else 0.0
        conf_std = np.mean(info["confs_std"]) if info["confs_std"] else 0.0
        
        if conf_min < aspect_conf_thresh:
            continue
        
        # Check for conflicts
        if _is_value_conflicted(aspect, v_can, kept):
            continue
        
        # Get the best text representation
        text = max(info["texts"], key=lambda s: len(s))
        
        # Add to kept list
        kept.append({
            "aspect": aspect,
            "canon": v_can,
            "text": text,
            "conf_mean": conf_mean,
            "conf_min": conf_min,
            "conf_gmean": conf_gmean,
            "conf_std": conf_std,
            "support": vote,
            "start": np.mean(info["starts"]) if info["starts"] else 0.0,
            "cat": cat,
            "title_tokens": title_tokens
        })
    
    return kept
def filter_im_lieferumfang_false_positives(predictions, title, category):
    if category != 1:  # Still mainly a category 1 issue
        return predictions
    
    filtered = []
    im_lief_preds = [p for p in predictions if p.get('aspect') == 'Im_Lieferumfang_Enthalten']
    other_preds = [p for p in predictions if p.get('aspect') != 'Im_Lieferumfang_Enthalten']
    
    # Core valid items from annexure
    valid_core_items = {
        'bremsscheiben', 'bremsbeläge', 'beläge', 'scheiben',
        'set', 'satz', 'kit', 'kette', 'wasserpumpe', 'zahnriemen',
        'bremssattel', 'bremstrommel', 'bremsbacken'
    }
    
    # Invalid patterns
    invalid_terms = {
        'für', 'fuer', 'passend', 'geeignet', 'kompatibel',
        'vorne', 'hinten', 'links', 'rechts', 'oben', 'unten',
        'original', 'neu', 'komplett'  # These are descriptors, not items
    }
    
    # Track what we're keeping
    kept_items = []
    
    for pred in im_lief_preds:
        value = pred.get('canon', '').lower()
        text = pred.get('text', '').lower()
        conf = pred.get('conf_mean', 0.0)
        
        # Rule 1: Skip obvious false positives
        if value in invalid_terms:
            continue
            
        # Rule 2: Multi-token validation (like "Beläge Scheiben")
        if ' ' in text:
            tokens = text.split()
            # Must contain at least one core item
            has_core = any(token in valid_core_items for token in tokens)
            # Must not be all descriptors
            all_invalid = all(token in invalid_terms for token in tokens)
            
            if not has_core or all_invalid:
                continue
                
            # Check for duplicate/subset predictions
            is_subset = any(value in kept for kept in kept_items)
            is_superset = any(kept in value for kept in kept_items)
            if is_subset or is_superset:
                # Keep the longer one if confidence is similar
                if is_subset and conf > 0.85:
                    kept_items = [k for k in kept_items if k not in value]
                else:
                    continue
        
        # Rule 3: Single token must be substantial
        elif len(value) <= 3:
            if value not in {'set', 'kit', 'öl'}:  # Only these short values are valid
                continue
        
        # Rule 4: Require very high confidence for generic terms
        generic_terms = {'set', 'satz', 'kit', 'teil', 'teile'}
        if value in generic_terms and conf < 0.93:
            continue
            
        # Rule 5: Numbers alone are not items
        if value.replace(' ', '').replace('-', '').isdigit():
            continue
            
        # Rule 6: Check title context for low-confidence predictions
        if conf < 0.90:
            # Item should appear in title
            if not any(item in title.lower() for item in value.split()):
                continue
        
        filtered.append(pred)
        kept_items.append(value)
    
    # Limit to max 3 items per title (from annexure examples)
    if len(filtered) > 3:
        filtered = sorted(filtered, key=lambda x: x.get('conf_mean', 0.0), reverse=True)[:3]
    
    return other_preds + filtered
def merge_split_im_lieferumfang(predictions):
    """
    Merge items that should be compound (e.g., 'Beläge' + 'Scheiben' -> 'Beläge Scheiben')
    Based on annexure patterns
    """
    merged = []
    skip_indices = set()
    
    # Common compound patterns from annexure
    compound_patterns = [
        ('beläge', 'scheiben'),
        ('bremsbeläge', 'bremsscheiben'),
        ('set', 'kette'),
        ('wasserpumpe', 'zahnriemensatz')
    ]
    
    for i, pred in enumerate(predictions):
        if i in skip_indices:
            continue
            
        if pred['aspect'] == 'Im_Lieferumfang_Enthalten':
            # Check next prediction
            if i + 1 < len(predictions):
                next_pred = predictions[i + 1]
                if next_pred['aspect'] == 'Im_Lieferumfang_Enthalten':
                    val1 = pred['canon'].lower()
                    val2 = next_pred['canon'].lower()
                    
                    # Check if they form a known compound
                    for p1, p2 in compound_patterns:
                        if (val1 == p1 and val2 == p2) or (val1 == p2 and val2 == p1):
                            # Merge them
                            merged_pred = dict(pred)
                            merged_pred['text'] = f"{pred['text']} {next_pred['text']}"
                            merged_pred['canon'] = f"{pred['canon']} {next_pred['canon']}"
                            merged_pred['conf_mean'] = min(pred['conf_mean'], next_pred['conf_mean'])
                            merged_pred['conf_min'] = min(pred['conf_min'], next_pred['conf_min'])
                            merged.append(merged_pred)
                            skip_indices.add(i + 1)
                            break
                    else:
                        merged.append(pred)
                else:
                    merged.append(pred)
            else:
                merged.append(pred)
        else:
            merged.append(pred)
    
    return merged
def frequency_aware_filtering(predictions, all_predictions_dict):
    """
    Filter predictions based on global frequency.
    High-frequency values need higher confidence.
    """
    # Count global frequencies for Im_Lieferumfang_Enthalten
    im_lief_frequencies = {}
    for record_preds in all_predictions_dict.values():
        for pred in record_preds:
            if pred.get('aspect') == 'Im_Lieferumfang_Enthalten':
                val = pred.get('canon', '').lower()
                im_lief_frequencies[val] = im_lief_frequencies.get(val, 0) + 1
    
    filtered = []
    for pred in predictions:
        aspect = pred.get('aspect')
        
        if aspect == 'Im_Lieferumfang_Enthalten':
            val = pred.get('canon', '').lower()
            freq = im_lief_frequencies.get(val, 0)
            conf = pred.get('conf_min', 0.0)
            
            # If this value appears >3000 times globally, need 99.5% confidence
            if freq > 2500 and conf < 0.996:
                continue  # Skip this prediction
            # If appears >2000 times, need 99.2% confidence  
            elif freq > 1500 and conf < 0.993:
                continue
            elif freq > 1000 and conf < 0.990:
                continue
                
        # Filter invalid Anzahl_Der_Einheiten patterns
        elif aspect == 'Anzahl_Der_Einheiten':
            value = pred.get('text', '')
            # Skip invalid patterns
            if any(x in value for x in ['mm', '(', ')', 'Satz', 'Set', 'Ø']):
                continue
            # Only keep proper number patterns
            if not re.match(r'^\d+[xX]?$|^\d+\s*[xX]$|^\d+Pcs$|^\d+-teilig$', value):
                continue
                
        filtered.append(pred)
    
    return filtered
def integrate_rare_aspects_into_predictions(predictions, title, category):
    # Get rare aspect predictions
    rare_preds = extract_rare_aspects(title, category)
    
    # Convert to your prediction format
    for aspect, value in rare_preds:
        # Check if this aspect-value pair already exists
        exists = False
        for pred in predictions:
            if pred.get('aspect') == aspect and canonicalize_for_match(pred.get('canon', '')) == canonicalize_for_match(value):
                exists = True
                break
        
        if not exists:
            # Add with high confidence since these are rule-based
            predictions.append({
                'aspect': aspect,
                'canon': value,
                'text': value,
                'conf_mean': 0.85,  # High confidence for rule-based
                'conf_min': 0.85,
                'conf_gmean': 0.85,
                'conf_std': 0.0,
                'support': 1.0,
                'start': 0,  # Position not critical for these
                'cat': category
            })
    
    return predictions
# =================== OOF store & calibration ======================
def build_oof_store():
    store = []
    for fold_k, (train_idx, val_idx) in enumerate(fold_splits):
        models_oof = [m for j,m in enumerate(fold_models) if j != fold_k]  # 4-model ensemble
        n_models = len(models_oof)
        for i in val_idx:
            ex = records[i]
            tokens, cat = ex["tokens"], ex["cat"]
            groups = _build_groups_for_models(models_oof, tokens, cat)
            true_pairs = {(a, canonicalize_for_match(v)) for (a,v) in decode_entities(tokens, ex["bio"])}
            store.append({"cat":cat, "title_tokens":tokens, "groups":groups,
                          "n_models":n_models, "true":true_pairs})
    return store

def pairs_by_cat_aspect(store, vote_fraction, conf_thresh):
    true_by = defaultdict(list)
    pred_by = defaultdict(list)
    for item in store:
        kept = strict_select_from_groups(item["groups"], item["cat"], item["title_tokens"],
                                         item["n_models"], vote_fraction, conf_thresh)
        preds = [(d["aspect"], d["canon"]) for d in kept]
        trues = list(item["true"])
        for a,v in preds:
            pred_by[(item["cat"], a)].append((a,v))
        for a,v in trues:
            true_by[(item["cat"], a)].append((a,v))
    return true_by, pred_by

def per_aspect_f02(true_list, pred_list):
    t_count, p_count = Counter(true_list), Counter(pred_list)
    tp = 0
    for k, c in p_count.items():
        if k in t_count: tp += min(c, t_count[k])
    p = tp/(sum(p_count.values())+1e-12)
    r = tp/(sum(t_count.values())+1e-12)
    return fbeta(p, r, 0.2)

def eval_comp_metric(store, vote_fraction, conf_thresh):
    true_by, pred_by = pairs_by_cat_aspect(store, vote_fraction, conf_thresh)
    cat_scores = []
    for cat in [1,2]:
        num, den = 0.0, 0.0
        for a in sorted(ALLOWED_ASP[cat]):
            t = true_by.get((cat, a), [])
            if len(t) == 0: continue
            p = pred_by.get((cat, a), [])
            f = per_aspect_f02(t, p)
            w = float(len(t))
            num += w * f
            den += w
        cat_scores.append((num/den) if den>0 else 0.0)
    overall = sum(cat_scores)/len(cat_scores)
    all_t = []; all_p = []
    for k in true_by: all_t.extend(true_by[k])
    for k in pred_by: all_p.extend(pred_by[k])
    p_m, r_m, f_m = compute_entity_f02(all_t, all_p)
    ok = True
    for cat in [1,2]:
        t_cat = []; p_cat = []
        for a in ALLOWED_ASP[cat]:
            t_cat.extend(true_by.get((cat,a), []))
            p_cat.extend(pred_by.get((cat,a), []))
        _, r_cat, _ = compute_entity_f02(t_cat, p_cat)
        if r_cat < CALIB_RECALL_FLOOR_CAT[cat]: ok = False
    if not ok:
        overall = 0.0
    return overall, (p_m, r_m, f_m)

def coordinate_descent_calibration_cataware(store, base_vote_fraction, base_conf_thresh,
                                            iters=CALIB_ITERS,
                                            conf_steps=CALIB_CONF_STEPS,
                                            vote_steps=CALIB_VOTE_STEPS):
    PREC_FLOOR_DROP = CALIB_PREC_FLOOR_DROP
    EPS = 1e-5


    def _init_maps(base_vf, base_ct):
        vf, ct = {}, {}
        for c in [1, 2]:
            for a in sorted(ALLOWED_ASP[c]):
                vf[(c, a)] = base_vf.get((c, a), base_vf.get(a, 0.6))
                ct[(c, a)] = base_ct.get((c, a), base_ct.get(a, 0.55))
        return vf, ct

    vf, ct = _init_maps(base_vote_fraction, base_conf_thresh)

    def candidates_around(val, steps, lo, hi):
        vals = {val}
        for s in steps:
            vals.add(max(lo, min(hi, val + s)))
            vals.add(max(lo, min(hi, val - s)))
        return sorted(vals)

    def aspect_keys_by_weight(vf, ct):
        true_by, _ = pairs_by_cat_aspect(store, vf, ct)
        items = []
        for c in [1,2]:
            for a in sorted(ALLOWED_ASP[c]):
                w = len(true_by.get((c,a), []))
                if w > 0:
                    items.append(((c,a), w))
        items.sort(key=lambda x: x[1], reverse=True)  # å¤§æƒé‡ä¼˜å…ˆ
        return [k for k,_ in items]

    start_overall, (p_start, r_start, f_start) = eval_comp_metric(store, vf, ct)
    best_overall = start_overall
    best_vf, best_ct = dict(vf), dict(ct)
    print(f"[CALIB] start   CompF0.2={start_overall:.4f} | micro P={p_start:.4f} R={r_start:.4f} F0.2={f_start:.4f}")

    conf_bias_cands = [0.0, 0.04, 0.08, -0.04]
    vote_bias_cands = [0.0, 0.08, 0.12, -0.08]
    for db in conf_bias_cands:
        for dv in vote_bias_cands:
            vf2, ct2 = dict(vf), dict(ct)
            for c in [1,2]:
                for a in sorted(ALLOWED_ASP[c]):
                    vf2[(c,a)] = max(0.20, min(1.00, vf[(c,a)] + dv))
                    ct2[(c,a)] = max(0.50, min(0.98, ct[(c,a)] + db))
            overall, (p_m, _, _) = eval_comp_metric(store, vf2, ct2)
            if p_m + 1e-12 < (p_start - PREC_FLOOR_DROP):
                continue
            if overall > best_overall + EPS:
                best_overall = overall
                best_vf, best_ct = dict(vf2), dict(ct2)
    vf, ct = dict(best_vf), dict(best_ct)

    for it in range(1, iters+1):
        improved = False
        order = aspect_keys_by_weight(vf, ct)  # æŒ‰æƒé‡ä»Žå¤§åˆ°å°

        for (c,a) in order:
            base = ct[(c,a)]
            tried = candidates_around(base, conf_steps, lo=0.50, hi=0.98)
            local_best = (best_overall, base)
            for val in tried:
                ct[(c,a)] = val
                overall, (p_m, _, _) = eval_comp_metric(store, vf, ct)
                if p_m + 1e-12 < (p_start - PREC_FLOOR_DROP):
                    continue
                if overall > local_best[0] + EPS:
                    local_best = (overall, val)
            if local_best[1] != base:
                ct[(c,a)] = local_best[1]
                best_overall = local_best[0]
                improved = True
            else:
                ct[(c,a)] = base

        for (c,a) in order:
            base = vf[(c,a)]
            tried = candidates_around(base, vote_steps, lo=0.20, hi=1.00)
            local_best = (best_overall, base)
            for val in tried:
                vf[(c,a)] = val
                overall, (p_m, _, _) = eval_comp_metric(store, vf, ct)
                if p_m + 1e-12 < (p_start - PREC_FLOOR_DROP):
                    continue
                if overall > local_best[0] + EPS:
                    local_best = (overall, val)
            if local_best[1] != base:
                vf[(c,a)] = local_best[1]
                best_overall = local_best[0]
                improved = True
            else:
                vf[(c,a)] = base

        _, (p_m, r_m, f_m) = eval_comp_metric(store, vf, ct)
        print(f"[CALIB] iter {it} CompF0.2={best_overall:.4f} | micro P={p_m:.4f} R={r_m:.4f} F0.2={f_m:.4f}")

        if improved:
            best_vf, best_ct = dict(vf), dict(ct)
        else:
            break

    _, (p_m, r_m, f_m) = eval_comp_metric(store, best_vf, best_ct)
    return best_vf, best_ct, (best_overall, (p_m, r_m, f_m))



# =================== Stage-2 verifier (same, with PN char-LR) ==================
def _shape_feats(text):
    s = text
    n = len(s) + 1e-6
    alpha = sum(ch.isalpha() for ch in s)/n
    digit = sum(ch.isdigit() for ch in s)/n
    punc  = sum((not ch.isalnum() and not ch.isspace()) for ch in s)/n
    caps  = sum(ch.isupper() for ch in s)/max(1.0, sum(ch.isalpha() for ch in s))
    return alpha, digit, punc, caps
def _best_gaz_sim_with_fuzzy(aspect, canon_val):
    score, _ = enhanced_gaz_fuzzy_match(aspect, canon_val, threshold=0.6)
    return float(score)
def _best_gaz_sim(aspect, canon_val):
    """
    Fallback for the original _best_gaz_sim function
    This wraps the fuzzy matching version
    """
    return _best_gaz_sim_with_fuzzy(aspect, canon_val)
def enhanced_gaz_fuzzy_match(aspect, text, threshold=0.75):
    """
    Returns (best_match_score, matched_value)
    """
    if aspect not in GAZ_ORIG:
        return 0.0, None
    
    text_clean = canonicalize_for_match(text)
    
    # Quick exact match check
    if text_clean in GAZ_CANON.get(aspect, set()):
        return 1.0, text_clean
    
    # German-specific replacements for better matching
    german_replacements = [
        ('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'),
('ß', 'ss'), ('&', 'und')
    ]
    
    text_variants = [text_clean]
    for old, new in german_replacements:
        text_variants.append(text_clean.replace(old, new))
        text_variants.append(text_clean.replace(new, old))
    
    best_score = 0.0
    best_match = None
    
    gaz_values = list(GAZ_ORIG.get(aspect, []))
    
    # Optimize by sampling for large gazetteers
    if len(gaz_values) > 100:
        # Sample more densely
        import random
        sample_size = min(100, len(gaz_values))
        gaz_values = random.sample(gaz_values, sample_size)
    
    for gaz_val in gaz_values:
        gaz_clean = canonicalize_for_match(gaz_val)
        
        for text_var in text_variants:
            # Standard sequence matching
            score = SequenceMatcher(None, text_var, gaz_clean).ratio()
            
            # Boost score for prefix matches (common in German compounds)
            if text_var.startswith(gaz_clean[:min(4, len(gaz_clean))]):
                score += 0.1
            
            # Boost for suffix matches (part numbers often)
            if text_var.endswith(gaz_clean[-min(4, len(gaz_clean)):]):
                score += 0.1
            
            # Acronym matching
            if aspect in ["Kompatible_Fahrzeug_Marke", "Hersteller"]:
                if len(gaz_clean) <= 4 and text_var == gaz_clean:
                    score = 1.0
            
            # Number pattern matching for part numbers
            if aspect in ["Herstellernummer", "Oe/Oem_Referenznummer(N)"]:
                # Extract number patterns
                text_nums = re.findall(r'\d+', text_var)
                gaz_nums = re.findall(r'\d+', gaz_clean)
                if text_nums and gaz_nums:
                    if text_nums == gaz_nums:
                        score = max(score, 0.9)
                    elif any(n in gaz_nums for n in text_nums):
                        score = max(score, 0.7)
            
            score = min(score, 1.0)
            
            if score > best_score:
                best_score = score
                best_match = gaz_val
    
    if best_score >= threshold:
        return best_score, best_match
    return best_score, None

def _ctx_hash(token):
    return hash(token) % 9973

CHARPN_VEC = None
CHARPN_LR  = None
def train_charpn_from_oof(oof_store):
    global CHARPN_VEC, CHARPN_LR
    texts, labels = [], []
    seen = set()
    for item in oof_store:
        for (a, v_can), info in item["groups"].items():
            if a not in PN_ASPECTS: continue
            text_out = max(info["texts"], key=lambda s: len(s))
            key = (a, v_can, text_out)
            if key in seen: continue
            seen.add(key)
            label = int((a, v_can) in item["true"])
            texts.append(text_out)
            labels.append(label)
    if len(set(labels)) < 2 or len(texts) < 100:
        CHARPN_VEC, CHARPN_LR = None, None
        print("[CHARPN] Not enough data to train char-level PN scorer.")
        return
    CHARPN_VEC = TfidfVectorizer(analyzer="char", ngram_range=(2,6), min_df=2)
    X = CHARPN_VEC.fit_transform(texts)
    CHARPN_LR = LogisticRegression(solver="saga", penalty="l2", C=4.0, max_iter=300, random_state=SEED)
    CHARPN_LR.fit(X, np.asarray(labels))
    print(f"[CHARPN] Trained on {len(texts)} PN candidates.")

def charpn_proba(text: str) -> float:
    if CHARPN_VEC is None or CHARPN_LR is None:
        return 0.5
    X = CHARPN_VEC.transform([text])
    return float(CHARPN_LR.predict_proba(X)[:,1][0])

def _make_features(d):
    a = d["aspect"]; text = d["text"]; cat = d["cat"]; v_can = d["canon"]
    conf_min  = d.get("conf_min", 0.0)
    conf_mean = d.get("conf_mean", 0.0)
    conf_gm   = d.get("conf_gmean", 0.0)
    conf_std  = d.get("conf_std", 0.0)
    support   = float(d.get("support", 0))
    seen_train = float(TRAIN_VALUE_FREQ[a][v_can] > 0)
    seen_train_log = math.log1p(TRAIN_VALUE_FREQ[a][v_can])
    alpha, digit, punc, caps = _shape_feats(text)
    n_tok = float(len(text.split()))
    n_chr = float(len(text))
    start = float(d.get("start", 0.0))
    title_len = max(1.0, len(d["title_tokens"]))
    pos_norm = start / title_len
    gaz_sim = _best_gaz_sim(a, v_can)
    in_gaz = 1.0 if (a in GAZ_CANON and v_can in GAZ_CANON[a]) else 0.0
    idx = int(start)
    left = canonicalize_for_match(d["title_tokens"][idx-1]) if idx-1 >= 0 else "<BOS>"
    right= canonicalize_for_match(d["title_tokens"][min(idx+len(text.split()), len(d['title_tokens'])-1)]) if len(d["title_tokens"])>0 else "<EOS>"
    left_h = _ctx_hash(left); right_h = _ctx_hash(right)
    pn_p = charpn_proba(text) if a in PN_ASPECTS else 0.5
    pa_generic = 0.0
    pa_head    = 0.0
    pa_cross   = 0.0
    if a == "Produktart":
        pa_generic = 1.0 if _produktart_is_generic_only(text) else 0.0
        pa_head    = 1.0 if _produktart_has_head(text, cat) else 0.0
        pa_cross   = 1.0 if _produktart_cross_banned(text, cat) else 0.0


    # --- new features to help KFM verifier ---
    is_pn_shape = 1.0 if _is_pnish(text) else 0.0
    brand_near = 0.0
    if a == "Kompatibles_Fahrzeug_Modell":
        bpos = _brand_token_positions(d["title_tokens"])
        if bpos:
            brand_near = 1.0 if min(abs(int(d.get("start",0)) - bp) for bp in bpos) <= 5 else 0.0

    return np.asarray([
        support, conf_min, conf_mean, conf_gm, conf_std,
        seen_train, seen_train_log,
        n_tok, n_chr, pos_norm,
        alpha, digit, punc, caps,
        gaz_sim, in_gaz,
        float(left_h), float(right_h),
        float(cat),
        pn_p,
        # new:
        is_pn_shape,
        brand_near,
        pa_generic, pa_head, pa_cross,
    ], dtype=np.float32)


def train_stage2_verifier(oof_store, vote_fraction, conf_thresh):
    train_charpn_from_oof(oof_store)

    X_by_key, y_by_key = defaultdict(list), defaultdict(list)
    base_pred_by, base_true_by = defaultdict(list), defaultdict(list)

    print("\n[VERIFIER] Building OOF set from stage-1 calibrated decisions...")
    for item in oof_store:
        kept = strict_select_from_groups(item["groups"], item["cat"], item["title_tokens"],
                                         item["n_models"], vote_fraction, conf_thresh)
        true_set = item["true"]
        for d in kept:
            key = (item["cat"], d["aspect"])
            X_by_key[key].append(_make_features(d))
            y_by_key[key].append(1 if (d["aspect"], d["canon"]) in true_set else 0)
            base_pred_by[(item["cat"], d["aspect"])].append((d["aspect"], d["canon"]))
        for a,v in true_set:
            base_true_by[(item["cat"], a)].append((a,v))

    def compF(true_by, pred_by):
        cat_scores=[]
        for c in [1,2]:
            num=0.0; den=0.0
            for a in ALLOWED_ASP[c]:
                t = true_by.get((c,a), [])
                if not t: continue
                p = pred_by.get((c,a), [])
                f = per_aspect_f02(t,p); w=float(len(t))
                num += w*f; den += w
            cat_scores.append((num/den) if den>0 else 0.0)
        return sum(cat_scores)/len(cat_scores)

# Print Stage-1 OOF using the exact same evaluator as the calibrator
    calib_compF, (calib_P, calib_R, calib_F02) = eval_comp_metric(oof_store, vote_fraction, conf_thresh)
    print(f"[VERIFIER] Stage-1 OOF  CompF0.2={calib_compF:.4f} | micro P={calib_P:.4f} R={calib_R:.4f} F0.2={calib_F02:.4f}")


    models, taus = {}, {}
    for c in [1,2]:
        for a in sorted(ALLOWED_ASP[c]):
            key = (c,a)
            X = np.asarray(X_by_key[key], dtype=np.float32) if len(X_by_key[key]) else None
            y = np.asarray(y_by_key[key], dtype=np.int32) if len(y_by_key[key]) else None
            if X is None or len(X)==0 or len(np.unique(y))<2: continue
            clf = HistGradientBoostingClassifier(
                loss="log_loss", max_iter=350, learning_rate=0.07,
                max_depth=None, max_leaf_nodes=31, random_state=SEED
            )
            clf.fit(X, y)
            models[key] = clf

            proba = clf.predict_proba(X)[:,1]
            qs = np.unique(np.quantile(proba, np.linspace(0.0, 1.0, VER_QUANT_STEPS)))
            best_f, best_tau = -1.0, 0.5
            pos = (y==1)
            for tau in qs:
                keep = (proba >= tau)
                tp = float(np.sum(keep & pos))
                fp = float(np.sum(keep & (~pos)))
                fn = float(np.sum((~keep) & pos))
                p = tp / (tp+fp+1e-12); r = tp / (tp+fn+1e-12)
                if p < VER_PREC_FLOOR:           # <-- NEW: precision floor
                    continue
                b2 = 0.2*0.2
                f = (1+b2)*p*r/(b2*p + r + 1e-12) if (p+r)>0 else 0.0
                if r >= VER_RECALL_FLOOR and f > best_f:
                    best_f, best_tau = f, float(tau)
            taus[key] = best_tau

    ver_pred_by, ver_true_by = defaultdict(list), defaultdict(list)
    for item in oof_store:
        kept = strict_select_from_groups(item["groups"], item["cat"], item["title_tokens"],
                                         item["n_models"], vote_fraction, conf_thresh)
        filtered = []
        for d in kept:
            key = (item["cat"], d["aspect"])
            clf = models.get(key, None)
            if clf is None:
                filtered.append(d); continue
            p = float(clf.predict_proba(_make_features(d).reshape(1,-1))[:,1][0])
            if p >= taus.get(key, 0.5):
                filtered.append(d)
        for d in filtered:
            ver_pred_by[(item["cat"], d["aspect"])].append((d["aspect"], d["canon"]))
        for a,v in item["true"]:
            ver_true_by[(item["cat"], a)].append((a,v))

    def compF(true_by, pred_by):
        cat_scores=[]
        for c in [1,2]:
            num=0.0; den=0.0
            for a in ALLOWED_ASP[c]:
                t = true_by.get((c,a), [])
                if not t: continue
                p = pred_by.get((c,a), [])
                f = per_aspect_f02(t,p); w=float(len(t))
                num += w*f; den += w
            cat_scores.append((num/den) if den>0 else 0.0)
        return sum(cat_scores)/len(cat_scores)

    ver_compF = compF(ver_true_by, ver_pred_by)
    all_t = []; all_p = []
    for k in ver_true_by: all_t.extend(ver_true_by[k])
    for k in ver_pred_by: all_p.extend(ver_pred_by[k])
    p_m2, r_m2, f_m2 = compute_entity_f02(all_t, all_p)
    print(f"[VERIFIER] Stage-2 OOF  CompF0.2={ver_compF:.4f} | micro P={p_m2:.4f} R={r_m2:.4f} F0.2={f_m2:.4f}")

    use_it = USE_STAGE2_VERIFIER and (ver_compF >= max(calib_compF, VER_SAFE_FLOOR) - 1e-4)
    if not use_it:
        print("[VERIFIER] Safety -> DISABLED (use stage-1 only).")
        return {}, {}, False, (calib_compF, (calib_P, calib_R, calib_F02))
    print("[VERIFIER] ENABLED.")
    return models, taus, True, (ver_compF, (p_m2, r_m2, f_m2))

# =========================== Run calibration ======================
print("\n[STRICT OOF] Building OOF candidate store...")
oof_store = build_oof_store()

seed_vote_fraction = {a: (VOTE_THRESH_5_STRICT.get(a, VOTE_THRESH_5[a]) / 5.0) for a in ASPECT_TO_CATS}
seed_conf_thresh = dict(CONF_THRESH_STRICT)

print("[CALIB] Tuning stage-1 thresholds (contest metric)...")
final_vote_fraction, final_conf_thresh, (compF1, micro1) = coordinate_descent_calibration_cataware(
    oof_store, seed_vote_fraction, seed_conf_thresh,
    iters=CALIB_ITERS,
    conf_steps=CALIB_CONF_STEPS,
    vote_steps=CALIB_VOTE_STEPS
)
# After calibration (around line 2980), add:
print("\n[DEBUG] Calibration chose these thresholds:")
for aspect in ["Im_Lieferumfang_Enthalten", "Kompatible_Fahrzeug_Marke", "Produktart"]:
    for cat in [1, 2]:
        key = (cat, aspect)
        if key in final_conf_thresh:
            print(f"  {aspect} Cat{cat}: {final_conf_thresh[key]:.3f}")
print("\n[DEBUG] All OTHER calibrated thresholds:")
aspects_to_check = ["Hersteller", "Einbauposition", "Anzahl_Der_Einheiten", 
                    "Bremsscheiben-Aussendurchmesser", "Herstellernummer", 
                    "Bremsscheibenart", "Material", "Farbe", "Länge", "Breite"]
for aspect in aspects_to_check:
    for cat in [1, 2]:
        key = (cat, aspect)
        if key in final_conf_thresh:
            print(f"  {aspect} Cat{cat}: {final_conf_thresh[key]:.3f}")
# Override calibration for problematic aspects
overrides = {
    "Im_Lieferumfang_Enthalten": 0.99,
    "Anzahl_Der_Einheiten": 0.93,
    "Bremsscheiben-Aussendurchmesser": 0.91
}

for aspect, threshold in overrides.items():
    for cat in [1, 2]:
        key = (cat, aspect)
        if key in final_conf_thresh:
            final_conf_thresh[key] = threshold
    final_conf_thresh[aspect] = threshold
    
print(f"[OVERRIDE] Applied threshold overrides: {overrides}")
print("\n[DEBUG] Final thresholds for key aspects:")
for aspect in ["Im_Lieferumfang_Enthalten", "Kompatible_Fahrzeug_Marke", "Produktart"]:
    for cat in [1, 2]:
        key = (cat, aspect)
        if key in final_conf_thresh:
            print(f"  {aspect} Cat{cat}: {final_conf_thresh[key]:.3f}")

VER_MODELS, VER_TAUS, VER_ENABLED, (compF2, micro2) = train_stage2_verifier(
    oof_store, final_vote_fraction, final_conf_thresh
)

# =================== Inference (quiz) =============================
def _read_titles(path):
    if os.path.exists(path):
        return pd.read_csv(path, sep="\t", engine="python",
                           quoting=csv.QUOTE_MINIMAL, keep_default_na=False, on_bad_lines="skip")
    if os.path.exists(path + ".zip"):
        return pd.read_csv(path + ".zip", sep="\t", engine="python",
                           quoting=csv.QUOTE_MINIMAL, keep_default_na=False, on_bad_lines="skip",
                           compression="zip")
    raise FileNotFoundError(f"Could not locate titles at {path} or {path}.zip")

titles_df = _read_titles(TITLES_TSV)

if "Category Id" in titles_df.columns and "Category" not in titles_df.columns:
    titles_df = titles_df.rename(columns={"Category Id":"Category"})
titles_df = titles_df.rename(columns={titles_df.columns[0]:"Record Number",
                                      titles_df.columns[1]:"Category",
                                      titles_df.columns[2]:"Title"})
titles_df["Record Number"] = pd.to_numeric(titles_df["Record Number"], errors="coerce").astype("Int64")
titles_df["Category"]      = pd.to_numeric(titles_df["Category"], errors="coerce").astype("Int64")
titles_df["Title"]         = titles_df["Title"].astype(str)
titles_df = titles_df.dropna(subset=["Record Number","Category","Title"])

quiz = titles_df[(titles_df["Record Number"]>=5001) & (titles_df["Record Number"]<=30000)].copy()

def ensemble_title_stage1(title: str, cat_int: int, models=None):
    ms = fold_models if models is None else models
    tokens = title.split()
    group = _build_groups_for_models(ms, tokens, cat_int)
    
    for key, info in group.items():
        info["vote"] = len(info["folds"])
        info["conf_min"] = np.min(info["confs_min"]) if info["confs_min"] else 0.0
        info["conf_mean"] = np.mean(info["confs_mean"]) if info["confs_mean"] else 0.0
        info["conf_gmean"] = np.mean(info["confs_gmean"]) if info["confs_gmean"] else 0.0
        info["conf_std"] = np.mean(info["confs_std"]) if info["confs_std"] else 0.0
        info["start"] = np.mean(info["starts"]) if info["starts"] else 0.0
    
    kept = strict_select_from_groups_enhanced(group, int(cat_int), tokens,
                                             max(1, len(ms)), final_vote_fraction, final_conf_thresh)
    return kept


def apply_stage2_filter(cat, kept, title_tokens):
    if not VER_ENABLED:
        return _kept_after_caps(kept, cat, title_tokens)
    out = []
    for d in kept:
        key = (cat, d["aspect"])
        clf = VER_MODELS.get(key, None)
        if clf is None:
            out.append(d); continue
        p = float(clf.predict_proba(_make_features(d).reshape(1,-1))[:,1][0])
        if p >= VER_TAUS.get(key, 0.5):
            out.append(d)
    return _kept_after_caps(out, cat, title_tokens)
def _final_sanity_scrub(d):
    a, t = d["aspect"], canonicalize_for_match(d["text"])
    if a == "Kompatibles_Fahrzeug_Modell":
        if any(tok in KFM_BAN_TOK for tok in t.split()):
            return False
        # compute cues
        bpos = _brand_token_positions(d["title_tokens"])
        near_brand = bool(bpos) and (min(abs(int(round(d.get("start",0.0))) - bp) for bp in bpos) <= KFM_NEAR_BRAND_WIN)
        in_gaz = d["canon"] in GAZ_CANON.get("Kompatibles_Fahrzeug_Modell", set())
        if _kfm_veto(d["text"], d.get("start",0.0), d["title_tokens"],
                    allow_if_brand_and_gaz=True, in_gaz=in_gaz, near_brand=near_brand):
            return False

    if a in {"Herstellernummer","Oe/Oem_Referenznummer(N)"} and VOLT_RE.match(t):
        return False
    if a == "Maßeinheit" and t not in {"mm","cm","zoll","l"}:
        return False
    return True
# FIRST PASS - Build frequency cache
print("Building frequency cache...")
all_predictions_cache = {}
for _, row in quiz.iterrows():
    rid = int(row["Record Number"])
    cat = int(row["Category"])
    title = str(row["Title"])
    tokens = title.split()
    kept = ensemble_title_stage1(title, cat)
    all_predictions_cache[rid] = kept

# SECOND PASS - Process with all filters
rows = []
for _, row in quiz.iterrows():
    rid = int(row["Record Number"])
    cat = int(row["Category"])
    title = str(row["Title"])
    tokens = title.split()

    kept = ensemble_title_stage1(title, cat)
    final = apply_stage2_filter(cat, kept, tokens)

    # Add heuristics
    present = {(d["aspect"], d["canon"]) for d in final}
    for a, v in heuristic_pairs(cat, tokens, present):
        final.append({
            "aspect": a, 
            "canon": canonicalize_for_match(v), 
            "text": v,
            "conf_min": 0.99, 
            "support": 9, 
            "cat": cat, 
            "start": 0.0, 
            "title_tokens": tokens
        })

    # Apply caps
    final = apply_aspect_caps(final, cat, tokens)
    
    
    # Apply frequency filtering (but less aggressive on Im_Lief)
    final = frequency_aware_filtering(final, all_predictions_cache)
    final = merge_split_im_lieferumfang(final)
    title = str(row["Title"])
    final = filter_im_lieferumfang_false_positives(final, title, cat)
    
    # Conflict resolution
    final = conflict_resolver(cat, tokens, final)
    
    # Final checks
    final = [d for d in final if _final_sanity_scrub(d)]
    final = enforce_aspect_category_constraints(final)

    for d in final:
        rows.append([rid, d["cat"], d["aspect"], d["text"]])

rows.sort(key=lambda x: (x[0], x[2], x[3]))
with open(SUBMIT_OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
    w.writerow(["Record Number", "Category", "Aspect Name", "Aspect Value"])
    w.writerows(rows)

print(f"âœ” Wrote {len(rows)} predictions to {SUBMIT_OUT} | Stage2={'ON' if VER_ENABLED else 'OFF'}")
# ================= Optional strict-ensemble CV ====================
if RUN_STRICT_CV:
    print("\n[STRICT ENSEMBLE CV] Evaluating final pipeline on validation splits (contest metric).")
    P_comp = []
    for fold_k, (train_idx, val_idx) in enumerate(fold_splits):
        models_oof = [m for j,m in enumerate(fold_models) if j != fold_k]
        tmp_store=[]
        for i in val_idx:
            ex = records[i]
            tokens, cat = ex["tokens"], ex["cat"]
            groups = _build_groups_for_models(models_oof, tokens, cat)
            true_pairs = {(a, canonicalize_for_match(v)) for (a,v) in decode_entities(tokens, ex["bio"])}
            tmp_store.append({"cat":cat, "title_tokens":tokens, "groups":groups,
                              "n_models":len(models_oof), "true":true_pairs})
        compF, micro = eval_comp_metric(tmp_store, final_vote_fraction, final_conf_thresh)
        print(f"[Fold {fold_k+1}] CompF0.2={compF:.4f} | micro P={micro[0]:.4f} R={micro[1]:.4f} F0.2={micro[2]:.4f}")
        P_comp.append(compF)
    print(f"[STRICT ENSEMBLE CV] Mean CompF0.2={np.mean(P_comp):.4f}")