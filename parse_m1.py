import json
import glob
from pathlib import Path

files = glob.glob("artifacts/kaggle-2026-09-01-milestone-1/*.json")

print("=" * 80)
print("          INSPEÇÃO DE ARTEFATOS DOWNSCALE - MILESTONE 1           ")
print("=" * 80)

for fpath in sorted(files):
    p = Path(fpath)
    print(f"\n📄 ARQUIVO: {p.name}")
    print("-" * 50)
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Imprimir o JSON formatado/resumido
            formatted = json.dumps(data, indent=2)
            # Se for muito grande, imprimir as primeiras 30 linhas
            lines = formatted.splitlines()
            if len(lines) > 35:
                print("\n".join(lines[:35]))
                print(f"   ... (+ {len(lines)-35} linhas ocultas)")
            else:
                print(formatted)
    except Exception as e:
        print(f"Erro ao ler {p.name}: {e}")

