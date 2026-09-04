import json
import glob
from pathlib import Path

def analyze_dataset():
    print("=" * 85)
    print("      KAGGLE DUAL-T4 (SM75 TP=1 vs TP=2) MULTI-GPU PERFORMANCE ANALYSIS      ")
    print("=" * 85)

    artifact_dirs = glob.glob("artifacts/*/")
    if not artifact_dirs:
        print("[!] Nenhum diretório 'artifacts/' encontrado.")
        return

    print(f"Diretórios de Artefatos Encontrados:")
    for d in artifact_dirs:
        print(f"  • {d}")
    print("=" * 85 + "\n")

    json_files = glob.glob("artifacts/**/*.json", recursive=True)
    records = []
    
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                label = data.get("label", Path(jf).stem)
                summary = data.get("summary", data)
                
                tp_size = data.get("tensor_parallel_size", 2 if "tp2" in label.lower() else 1)
                concurrency = data.get("concurrency", data.get("num_prompts", 1))
                
                throughput = summary.get("output_throughput_tok_s", summary.get("tokens_per_second", 0))
                ttft_ms = summary.get("ttft_p95_ms", summary.get("ttft_ms", 0))
                tpot_ms = summary.get("tpot_p95_ms", summary.get("tpot_ms", 0))
                
                records.append({
                    "file": Path(jf).name,
                    "label": label,
                    "tp": tp_size,
                    "concurrency": concurrency,
                    "throughput": throughput,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms
                })
        except Exception:
            pass

    if records:
        print(f"{'Label / File':<35} | {'TP':<4} | {'Conc':<5} | {'Throughput (tok/s)':<20} | {'TTFT (ms)':<10}")
        print("-" * 85)
        for r in records:
            print(f"{r['label'][:35]:<35} | {r['tp']:<4} | {r['concurrency']:<5} | {r['throughput']:>18.2f} | {r['ttft_ms']:>10.2f}")
        print("=" * 85)
    else:
        print("Imprimindo arquivos de evidência em texto encontrados nos artefatos:\n")
        for txt_file in glob.glob("artifacts/**/*.txt", recursive=True) + glob.glob("artifacts/**/*.md", recursive=True):
            print(f"=== File: {txt_file} ===")
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                print(content[:1500]) # Primeiros 1500 caracteres
                print("\n" + "-"*80 + "\n")

if __name__ == "__main__":
    analyze_dataset()
