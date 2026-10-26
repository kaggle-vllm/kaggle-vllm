# Doctor and dependency inspection

`PYTHONPATH=src python3.13 -m kaggle_vllm.cli doctor --json` reports the active target's Python, Torch/CUDA, GPU topology, and provisional dependency checks. Before a cp313 native wheel is validated and registered, the result is intentionally incompatible even if the host matches the proposed hardware profile.

The active dependency specification keeps the historical vLLM 0.18.1 constraints visible, especially `transformers>=4.56,<5` and protobuf `>=5.29.6`. Kaggle's observed Transformers 5.16.1 and protobuf 5.29.5 conflict with those historical bounds. These checks diagnose contracts; they do not install packages or establish native compatibility.

The original cp312 overlay and dependency baseline are preserved under `compat/historical-py312-v0.2.0/` and are not loaded by the active profile.
