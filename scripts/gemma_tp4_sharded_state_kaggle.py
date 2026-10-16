"""Kaggle CPython 3.12, four L4, pinned vLLM 0.30.0 sharded_state trial.

Run only after the new-SDK acceptance and offline runtime bootstrap. This
script never downloads, uploads, or writes to a Kaggle Input. A complete
load-back and deterministic generation comparison are required before it
reports validation. The upstream vLLM 0.30.0 EngineCoreClient exports the
four worker state dictionaries through save_sharded_state().
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

MODEL_DEFAULT = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
OUTPUT_DEFAULT = Path('/kaggle/working/gemma-4-31b-it-qat-w4a16-ct-tp4-vllm030-sm89')
SHARD = re.compile(r'^model-rank-([0-3])-part-(\d+)\.safetensors$')
NATIVE_SHA = '9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8'
SDK_SHA = 'a14a0be361ea0ab876aa939743ad7259ac012b8c9410cc668899da00b875c27b'
PROMPT = 'Explain tensor parallelism in one sentence.'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_member(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise RuntimeError(f'unsafe manifest member: {name}')
    target = root / relative
    if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink() or not target.is_file():
        raise RuntimeError(f'unsafe or missing manifest member: {name}')
    return target


def gpu_snapshot() -> list[dict]:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30, check=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) != 4:
            raise RuntimeError(f'unexpected nvidia-smi row: {line}')
        rows.append({'name': fields[0], 'cc': fields[1], 'used_mib': int(fields[2]), 'total_mib': int(fields[3])})
    if len(rows) != 4 or any(x['name'] != 'NVIDIA L4' or x['cc'] != '8.9' for x in rows):
        raise RuntimeError(f'requires exactly four L4 SM89 GPUs: {rows}')
    return rows


def preflight(model: Path, output: Path, sdk_wheel: Path, *, new_save: bool) -> dict:
    if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 12) or sys.platform != 'linux' or os.uname().machine != 'x86_64':
        raise RuntimeError('requires Kaggle CPython 3.12 Linux x86_64')
    if not model.is_dir() or Path('/kaggle/input') not in model.resolve().parents:
        raise RuntimeError('model must be an attached read-only Kaggle Input directory')
    if output.exists() and output.is_symlink():
        raise RuntimeError('output symlink rejected')
    if Path('/kaggle/working') not in output.parent.resolve().parents and output.parent.resolve() != Path('/kaggle/working'):
        raise RuntimeError('output must be under /kaggle/working')
    if not (model / 'config.json').is_file():
        raise RuntimeError('source model config missing')
    card = model / 'README.md'
    if not card.is_file() or not re.search(r'^license:\s*apache-2\.0\s*$', card.read_text(encoding='utf-8'), re.MULTILINE | re.IGNORECASE):
        raise RuntimeError('source model card must declare Apache-2.0 before checkpoint export')
    if sdk_wheel.name != 'kaggle_vllm-0.3.1-py3-none-any.whl' or sha256(sdk_wheel) != SDK_SHA:
        raise RuntimeError('SDK wheel identity differs from the published 0.3.1 artifact')
    if any(path.is_symlink() for path in model.rglob('*')):
        raise RuntimeError('model Input contains a symlink; resolve a safe immutable Input')
    import torch
    import vllm
    import kaggle_vllm
    if torch.__version__ != '2.13.0+cu129' or torch.version.cuda != '12.9' or vllm.__version__ != '0.30.0':
        raise RuntimeError(f'wrong runtime: torch={torch.__version__}, CUDA={torch.version.cuda}, vllm={vllm.__version__}')
    if kaggle_vllm.__version__ != '0.3.1':
        raise RuntimeError(f'wrong SDK version: {kaggle_vllm.__version__}')
    gpus = gpu_snapshot()
    if any(row['used_mib'] != 0 for row in gpus):
        raise RuntimeError(f'requires an idle four-L4 session before model loading: {gpus}')
    source_weights = sorted(p for p in model.rglob('*') if p.is_file() and p.suffix in ('.safetensors', '.bin', '.pt'))
    source_bytes = sum(p.stat().st_size for p in source_weights)
    if source_bytes == 0:
        raise RuntimeError('no local source model weight files found')
    free = shutil.disk_usage('/kaggle/working').free
    required = max(24 * 1024**3, int(source_bytes * 1.3))
    if new_save and free < required:
        raise RuntimeError(f'insufficient working disk: {free} free, {required} required')
    return {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda, 'vllm': vllm.__version__, 'sdk': kaggle_vllm.__version__, 'sdk_origin': str(Path(kaggle_vllm.__file__).resolve()), 'sdk_wheel_sha256': SDK_SHA, 'gpus': gpus, 'source_model_license': 'apache-2.0', 'source_model_config_sha256': sha256(model / 'config.json'), 'source_model_weights': [{'path': str(p.relative_to(model)), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in source_weights], 'source_weight_bytes': source_bytes, 'working_free_bytes': free, 'required_bytes': required}


def copy_metadata(source: Path, target: Path) -> list[str]:
    copied = []
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise RuntimeError(f'model metadata symlink rejected: {path}')
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if path.suffix in ('.safetensors', '.bin', '.pt', '.gguf'):
            continue
        if path.name == 'model.safetensors.index.json':
            dest = target / 'SOURCE_MODEL_INDEX.json'
        else:
            dest = target / relative
        if path.stat().st_size > 256 * 1024**2:
            raise RuntimeError(f'unexpectedly large metadata file: {relative}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        copied.append(str(dest.relative_to(target)))
    if 'config.json' not in copied:
        raise RuntimeError('config metadata not copied')
    return copied


def inspect_shards(root: Path) -> dict:
    ranks: dict[int, dict[int, Path]] = {rank: {} for rank in range(4)}
    for path in root.iterdir():
        match = SHARD.fullmatch(path.name)
        if path.suffix == '.safetensors' and match is None:
            raise RuntimeError(f'unexpected safetensors file in shard output: {path.name}')
        if match:
            rank, part = map(int, match.groups())
            if part in ranks[rank] or path.is_symlink() or not path.is_file():
                raise RuntimeError(f'duplicate or unsafe shard: {path}')
            ranks[rank][part] = path
    entries = []
    for rank, parts in ranks.items():
        if not parts or sorted(parts) != list(range(len(parts))):
            raise RuntimeError(f'rank {rank} missing or noncontiguous parts: {sorted(parts)}')
        keys_seen = set()
        for part, path in sorted(parts.items()):
            # Safetensors headers contain shape, dtype and offsets. Reading the
            # header avoids loading multi-GiB tensor payloads into CPU memory.
            with path.open('rb') as stream:
                header_size = int.from_bytes(stream.read(8), 'little')
                if not 0 < header_size < min(path.stat().st_size - 8, 64 * 1024**2):
                    raise RuntimeError(f'invalid safetensors header: {path.name}')
                header = json.loads(stream.read(header_size))
            tensors = {k: {'shape': v['shape'], 'dtype': v['dtype']} for k, v in header.items() if k != '__metadata__'}
            keys = list(tensors)
            if not keys or keys_seen.intersection(keys):
                raise RuntimeError(f'empty/duplicate tensor keys in {path.name}')
            keys_seen.update(keys)
            entries.append({'path': path.name, 'rank': rank, 'part': part, 'bytes': path.stat().st_size, 'sha256': sha256(path), 'tensor_count': len(keys), 'tensors': tensors})
    return {'ranks': [0, 1, 2, 3], 'shards': entries, 'tensor_keys_by_rank': {str(r): sum(e['tensor_count'] for e in entries if e['rank'] == r) for r in ranks}}


def child_phase(phase: str, model: Path, output: Path) -> None:
    import psutil
    from vllm import LLM, SamplingParams
    from fresh_kaggle_gemma_v030 import child_snapshot, shutdown_and_verify
    baseline = child_snapshot(psutil.Process(), psutil)
    llm = None
    failure = None
    report = {'phase': phase, 'status': 'FAIL', 'tensor_parallel_size': 4}
    try:
        llm = LLM(model=str(model if phase == 'save' else output), tensor_parallel_size=4, dtype='bfloat16', gpu_memory_utilization=0.86, max_model_len=2048, load_format='auto' if phase == 'save' else 'sharded_state', enforce_eager=True, disable_custom_all_reduce=True, trust_remote_code=False)
        result = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=40, seed=0))
        tokens = list(result[0].outputs[0].token_ids)
        report['generated_tokens'] = len(tokens)
        report['token_ids'] = tokens
        report['text'] = result[0].outputs[0].text
        if not tokens or not report['text'].strip():
            raise RuntimeError(f'{phase} generated no semantic text')
        if phase == 'save':
            core = getattr(llm.llm_engine, 'engine_core', None)
            save = getattr(core, 'save_sharded_state', None)
            if not callable(save):
                raise RuntimeError('pinned vLLM EngineCoreClient.save_sharded_state unavailable')
            save(path=str(output), pattern='model-rank-{rank}-part-{part}.safetensors', max_size=5 * 1024**3)
            report['save_api'] = 'LLM.llm_engine.engine_core.save_sharded_state'
        report['status'] = 'PASS'
    except BaseException as exc:
        failure = exc
        report['error'] = repr(exc)
    finally:
        if llm is not None:
            try:
                class Adapter: pass
                adapter = Adapter(); adapter.upstream = llm
                cleanup = shutdown_and_verify(adapter, psutil.Process(), baseline, psutil)
                report['cleanup'] = cleanup
                if cleanup['status'] != 'PASS':
                    report['status'] = 'FAIL'
                    failure = failure or RuntimeError('worker cleanup failed')
            except BaseException as exc:
                report['status'] = 'FAIL'; report['cleanup_error'] = repr(exc)
                failure = failure or exc
        (output / f'{phase}_phase_report.json').write_text(json.dumps(report, indent=2) + '\n')
    if failure:
        raise failure


def run_child(script: Path, phase: str, model: Path, output: Path, log: Path) -> None:
    proc = subprocess.Popen([sys.executable, str(script), '--internal-phase', phase, '--model', str(model), '--output', str(output)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={**os.environ, 'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1'}, start_new_session=True)
    tail: deque[bytes] = deque()
    kept = 0
    total = 0
    limit = 8 * 1024 * 1024
    def drain() -> None:
        nonlocal kept, total
        assert proc.stdout is not None
        for block in iter(lambda: proc.stdout.read1(65536), b''):
            tail.append(block)
            kept += len(block)
            total += len(block)
            while kept > limit:
                kept -= len(tail.popleft())
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        code = proc.wait(timeout=2400)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
        code = -1
    reader.join(timeout=15)
    if reader.is_alive():
        raise RuntimeError(f'{phase} log stream did not close after child exit')
    log.write_bytes(f'[captured tail of {total} output bytes; limit {limit} bytes]\n'.encode() + b''.join(tail))
    if code:
        raise RuntimeError(f'{phase} phase exit {code}; inspect {log}')
    for _ in range(30):
        snapshot = gpu_snapshot()
        if all(row['used_mib'] == 0 for row in snapshot):
            return
        time.sleep(1)
    raise RuntimeError(f'{phase} left GPU memory allocated: {snapshot}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=MODEL_DEFAULT)
    parser.add_argument('--output', type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument('--sdk-wheel', type=Path, required=False)
    parser.add_argument('--recover-incomplete', action='store_true')
    parser.add_argument('--internal-phase', choices=('save', 'load'))
    args = parser.parse_args()
    model, output = args.model, args.output
    if args.internal_phase:
        child_phase(args.internal_phase, model, output)
        return
    if args.sdk_wheel is None:
        raise RuntimeError('--sdk-wheel must name the exact attached 0.3.1 wheel')
    staging = output.with_name(output.name + '.incomplete')
    if not output.exists() and staging.exists() and args.recover_incomplete:
        if not (staging / 'PREFLIGHT.json').is_file() or staging.is_symlink():
            raise RuntimeError(f'unsafe or unowned incomplete save: {staging}')
        shutil.rmtree(staging)
    info = preflight(model, output, args.sdk_wheel, new_save=not output.exists())
    if output.exists():
        manifest_path = output / 'SHARDED_STATE_MANIFEST.json'
        if not manifest_path.is_file():
            raise RuntimeError('output exists without complete manifest; choose a new path')
        manifest = json.loads(manifest_path.read_text())
        current = inspect_shards(output)
        if current['shards'] != manifest['shards']:
            raise RuntimeError('completed shard layout differs from manifest')
        for entry in manifest['shards']:
            if sha256(safe_member(output, entry['path'])) != entry['sha256']:
                raise RuntimeError(f'completed shard hash drift: {entry["path"]}')
        for name, digest in manifest['metadata_sha256'].items():
            if sha256(safe_member(output, name)) != digest:
                raise RuntimeError(f'completed metadata hash drift: {name}')
    else:
        if staging.exists():
            raise RuntimeError(f'incomplete previous save at {staging}; inspect it and pass --recover-incomplete to discard only this owned staging directory')
        staging.mkdir(parents=True)
        (staging / 'PREFLIGHT.json').write_text(json.dumps(info, indent=2) + '\n')
        run_child(Path(__file__).resolve(), 'save', model, staging, staging / 'SAVE.log')
        metadata = copy_metadata(model, staging)
        layout = inspect_shards(staging)
        manifest = {'schema_version': 1, 'status': 'SAVED_NOT_YET_READBACK', 'base_model': 'google/gemma-4-31B-it-qat-w4a16-ct', 'source_input': str(model), 'source_model_license': info['source_model_license'], 'source_model_config_sha256': info['source_model_config_sha256'], 'source_model_weights': info['source_model_weights'], 'sdk_wheel_sha256': SDK_SHA, 'native_wheel_sha256': NATIVE_SHA, 'vllm_version': '0.30.0', 'tensor_parallel_size': 4, 'gpu_arch': 'SM89', 'metadata_files': metadata, 'metadata_sha256': {name: sha256(staging / name) for name in metadata}, **layout}
        (staging / 'SHARDED_STATE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (staging / 'SHARDED_STATE_SHA256SUMS.txt').write_text(''.join(f'{x["sha256"]}  {x["path"]}\n' for x in layout['shards']))
        staging.rename(output)
    run_child(Path(__file__).resolve(), 'load', model, output, output / 'LOAD.log')
    saved = json.loads((output / 'save_phase_report.json').read_text())
    loaded = json.loads((output / 'load_phase_report.json').read_text())
    if saved.get('status') != 'PASS' or loaded.get('status') != 'PASS' or saved.get('token_ids') != loaded.get('token_ids'):
        raise RuntimeError('TP4 sharded-state reload did not reproduce the saved model generation')
    manifest['status'] = 'GEMMA_TP4_SHARDING_VALIDATED_LOCAL_KAGGLE'
    manifest['load_report'] = 'load_phase_report.json'
    manifest['equivalence'] = {'prompt': PROMPT, 'seed': 0, 'temperature': 0.0, 'token_ids_equal': True}
    manifest['postload_gpu'] = gpu_snapshot()
    temporary_manifest = output / 'SHARDED_STATE_MANIFEST.json.tmp'
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    os.replace(temporary_manifest, output / 'SHARDED_STATE_MANIFEST.json')
    print(json.dumps({'status': manifest['status'], 'output': str(output), 'shards': len(manifest['shards'])}))


if __name__ == '__main__':
    main()
