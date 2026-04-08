"""
test_pipeline.py — Executa o pipeline completo de segmentação ROQS
em um único sujeito, sem interface gráfica.

Uso:
    python test_pipeline.py -p /caminho/para/pasta_do_sujeito

A pasta deve conter os arquivos DTI (dti_L1/L2/L3, dti_V1/V2/V3)
em formato .nii.gz ou .nii.
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np

import warnings
warnings.filterwarnings('ignore')

# ── Argumento de entrada ─────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Testa o pipeline ROQS em um sujeito único.")
parser.add_argument('-p', '--path', required=True, help="Caminho para a pasta do sujeito")
args = parser.parse_args()

data_path = os.path.abspath(args.path)

if not os.path.isdir(data_path):
    print(f"[ERRO] Pasta não encontrada: {data_path}")
    sys.exit(1)

print("=" * 60)
print(f"  Sujeito : {data_path}")
print("=" * 60)

# ── Importações do pipeline ──────────────────────────────────────────────────

import segmentation as sg
import getParcellation as gm

# ── 1. Carregar dados DTI ────────────────────────────────────────────────────

print("\n[1/6] Carregando dados DTI...")
t0 = time.time()
try:
    wFA_v, FA_v, MD_v, RD_v, AD_v, fissure, eigvals, eigvects, affine = sg.run_analysis(data_path)
    print(f"      Volume shape  : {FA_v.shape}")
    print(f"      Fatia sagital : {fissure}")
    print(f"      OK ({time.time()-t0:.2f}s)")
except Exception as e:
    print(f"[ERRO] Falha ao carregar DTI: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── Extrair fatia sagital ────────────────────────────────────────────────────

wFA = wFA_v[fissure, :, :]
FA  = FA_v[fissure, :, :]
MD  = MD_v[fissure, :, :]
RD  = RD_v[fissure, :, :]
AD  = AD_v[fissure, :, :]
eigvects_ms = abs(eigvects[0, :, fissure])

# ── 2. Segmentação ROQS ──────────────────────────────────────────────────────

print("\n[2/6] Executando segmentação ROQS...")
t0 = time.time()
try:
    segmentation = sg.segm_roqs(wFA, eigvects_ms)
    n_pixels = int(np.sum(segmentation))
    print(f"      Pixels segmentados : {n_pixels}")
    print(f"      OK ({time.time()-t0:.2f}s)")
except Exception as e:
    print(f"[ERRO] Segmentação falhou: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 3. Escalares globais ─────────────────────────────────────────────────────

print("\n[3/6] Calculando escalares...")
try:
    scalar_statistics = sg.getScalars(segmentation, FA, MD, RD, AD)
    labels = ['FA', 'FA StdDev', 'MD', 'MD StdDev', 'RD', 'RD StdDev', 'AD', 'AD StdDev']
    for label, value in zip(labels, scalar_statistics):
        print(f"      {label:<12}: {value:.6f}")
except Exception as e:
    print(f"[ERRO] Escalares: {e}")
    traceback.print_exc()

# ── 4. Midlines ──────────────────────────────────────────────────────────────

print("\n[4/6] Calculando midlines (200 pontos)...")
try:
    scalar_midlines = {
        'FA': sg.getFAmidline(segmentation, FA,  n_points=200),
        'MD': sg.getFAmidline(segmentation, MD,  n_points=200),
        'RD': sg.getFAmidline(segmentation, RD,  n_points=200),
        'AD': sg.getFAmidline(segmentation, AD,  n_points=200),
    }
    for key, vals in scalar_midlines.items():
        arr = np.array(vals)
        print(f"      {key} — min: {arr.min():.6f}  max: {arr.max():.6f}  mean: {arr.mean():.6f}")
except Exception as e:
    print(f"[ERRO] Midlines: {e}")
    traceback.print_exc()
    scalar_midlines = {'FA': [], 'MD': [], 'RD': [], 'AD': []}

# ── 5. Espessura ─────────────────────────────────────────────────────────────

print("\n[5/6] Calculando espessura...")
try:
    col_heights = np.sum(segmentation, axis=0).astype(float)
    thickness = np.interp(
        np.linspace(0, max(len(col_heights) - 1, 1), 200),
        np.arange(len(col_heights)),
        col_heights
    )
    print(f"      min: {thickness.min():.2f}  max: {thickness.max():.2f}  mean: {thickness.mean():.2f} px")
except Exception as e:
    print(f"[ERRO] Espessura: {e}")
    traceback.print_exc()
    thickness = np.zeros(200)

# ── 6. Parcelamento ──────────────────────────────────────────────────────────

print("\n[6/6] Calculando parcelamento (Witelson, Hofer, Chao, Cover, Freesurfer)...")
try:
    scalar_maps = (FA, MD, RD, AD)
    values = gm.getParcellation(segmentation, FA)
    parcellation_dict = gm.parcellations_dfs_dicts(scalar_maps, values)

    for method in ['Witelson', 'Hofer', 'Chao', 'Cover', 'Freesurfer']:
        fas = [parcellation_dict[method][f'P{i+1}']['FA'] for i in range(5)]
        print(f"      {method:<12}: FA por região = " + "  ".join(f"P{i+1}={v:.4f}" for i, v in enumerate(fas)))
except Exception as e:
    print(f"[ERRO] Parcelamento: {e}")
    traceback.print_exc()

# ── Resumo final ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Pipeline concluído com sucesso.")
print("=" * 60)
