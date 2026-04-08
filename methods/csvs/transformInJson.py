# Bibliotecas
import ast
import json
import math
import re
import numpy as np
import pandas as pd


# ── Parsing de células com listas codificadas como string ────────────────────

def _parse_list_cell(val):
    """Converte uma célula CSV que contém uma string de lista Python em list."""
    if not isinstance(val, str):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        # Fallback: limpa repr numpy (np.float64(...)) e parseia como JSON
        try:
            cleaned = re.sub(r'np\.float\d+\(([^)]+)\)', r'\1', val)
            return json.loads(cleaned.replace("'", '"'))
        except Exception:
            return []


def dataFrameStringToList(df):
    """Aplica _parse_list_cell em todas as colunas usando apply (vetorizado)."""
    return df.apply(lambda col: col.map(_parse_list_cell))


# ── NaN check eficiente ───────────────────────────────────────────────────────

def _has_nan(subject):
    """Retorna True se o sujeito contém algum NaN nos campos numéricos."""
    for val in subject.values():
        if isinstance(val, dict):
            for v2 in val.values():
                if isinstance(v2, list):
                    try:
                        if np.any(np.isnan(np.array(v2, dtype=float))):
                            return True
                    except (TypeError, ValueError):
                        pass
                else:
                    try:
                        if math.isnan(v2):
                            return True
                    except (TypeError, ValueError):
                        pass
        elif isinstance(val, list):
            try:
                if np.any(np.isnan(np.array(val, dtype=float))):
                    return True
            except (TypeError, ValueError):
                pass
    return False


# ── Classe Subject ────────────────────────────────────────────────────────────

class Subject:
    def __init__(self, name, watershed_scalar, ROQS_scalars,
                 watershed_midlines, ROQS_midlines,
                 watershed_thickness, ROQS_thickness,
                 watershed_parcellation, ROQS_parcellation,
                 santarosa_scalars):
        self.name = self._adjust_name(str(name))
        self.watershed_scalar       = watershed_scalar
        self.ROQS_scalars           = ROQS_scalars
        self.watershed_midlines     = watershed_midlines
        self.ROQS_midlines          = ROQS_midlines
        self.watershed_thickness    = list(watershed_thickness)
        self.ROQS_thickness         = list(ROQS_thickness)
        self.watershed_parcellation = watershed_parcellation
        self.ROQS_parcellation      = ROQS_parcellation
        self.santarosa_scalars      = santarosa_scalars

    def _adjust_name(self, name):
        if name.startswith("Subject_"):
            name = name[len("Subject_"):]
        return name.zfill(7)

    def to_dict(self):
        return {
            "Id": self.name,
            "Watershed_scalar":    dict(self.watershed_scalar),
            "ROQS_scalar":         dict(self.ROQS_scalars),
            "santarosa_scalars":   dict(self.santarosa_scalars),
            "Watershed_midlines":  dict(self.watershed_midlines),
            "ROQS_midlines":       dict(self.ROQS_midlines),
            "Watershed_thickness": self.watershed_thickness,
            "ROQS_thickness":      self.ROQS_thickness,
            "Watershed_parcellation": dict(self.watershed_parcellation),
            "ROQS_parcellation":   dict(self.ROQS_parcellation),
        }


# ── Utilitários de leitura ────────────────────────────────────────────────────

def _safe_drop_index(df):
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    return df.drop(columns=unnamed) if unnamed else df


def _read_csv(filename, required=True):
    try:
        return _safe_drop_index(pd.read_csv(filename, sep=";"))
    except FileNotFoundError:
        if required:
            print(f"\n[ERRO] Arquivo não encontrado: {filename}")
            print("       Execute a análise ROQS antes de converter para JSON.")
            raise
        return pd.DataFrame()


# ── Leitura dos CSVs ──────────────────────────────────────────────────────────

ROQS_scalar      = _read_csv("ROQS_scalar_statistics.csv")
watershed_scalar = _read_csv("Watershed_scalar_statistics.csv")

try:
    santarosa_scalar = _read_csv("santarosa.csv", required=False)
    if santarosa_scalar.empty:
        santarosa_scalar = ROQS_scalar.copy()
except FileNotFoundError:
    santarosa_scalar = ROQS_scalar.copy()

# Midlines: parse string→list usando apply (bem mais rápido que loop manual)
ROQS_midlines      = dataFrameStringToList(_read_csv("ROQS_scalar_midlines.csv"))
watershed_midlines = dataFrameStringToList(_read_csv("Watershed_scalar_midlines.csv"))

ROQS_thickness      = _read_csv("ROQS_dict_thickness.csv")
watershed_thickness = _read_csv("Watershed_dict_thickness.csv")

ROQS_parcellation      = _read_csv("ROQS_parcellation_statistics.csv")
watershed_parcellation = _read_csv("Watershed_parcellation_statistics.csv")

names      = list(ROQS_parcellation["Name"])
n_santa    = len(santarosa_scalar)

# ── Construção dos sujeitos ───────────────────────────────────────────────────

subjects_list = []
for i, name in enumerate(names):
    santa_i = i % n_santa if n_santa > 0 else 0
    sub = Subject(
        name,
        watershed_scalar.iloc[i],
        ROQS_scalar.iloc[i],
        watershed_midlines.iloc[i],
        ROQS_midlines.iloc[i],
        watershed_thickness.iloc[i],
        ROQS_thickness.iloc[i],
        watershed_parcellation.iloc[i],
        ROQS_parcellation.iloc[i],
        santarosa_scalar.iloc[santa_i],
    )
    subjects_list.append(sub.to_dict())

# ── Remoção de sujeitos com NaN (passagem única) ──────────────────────────────

subjects_list = [s for s in subjects_list if not _has_nan(s)]

# ── Escrita do JSON ───────────────────────────────────────────────────────────

with open("../../src/data/mydata.json", "w") as f:
    json.dump(subjects_list, f)

print(f"[OK] {len(subjects_list)} sujeito(s) exportado(s) para mydata.json")
