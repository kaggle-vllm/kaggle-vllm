from kaggle_vllm.server import ServerConfig, build_server_command


def test_safe_qwen_sharded_server_command():
    config = ServerConfig(
        model="/kaggle/input/qwen2.5-3b-t4x2-sharded",
        model_revision="immutable",
        served_model_name="qwen2.5-3b-kaggle-t4x2",
        tensor_parallel_size=2,
        load_format="sharded_state",
        max_model_len=2048,
        port=8001,
        gpu_memory_utilization=0.7,
        max_num_batched_tokens=4096,
        max_num_seqs=64,
        seed=0,
    )
    command = build_server_command(
        config, executable="/staged/bin/vllm", validate_gpus=False
    )
    assert command[:3] == [
        "/staged/bin/vllm",
        "serve",
        "/kaggle/input/qwen2.5-3b-t4x2-sharded",
    ]
    assert command[command.index("--load-format") + 1] == "sharded_state"
    assert command[command.index("--revision") + 1] == "immutable"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert "--enforce-eager" in command
    assert "--disable-custom-all-reduce" in command
    assert command[command.index("--max-num-batched-tokens") + 1] == "4096"
    assert command[command.index("--max-num-seqs") + 1] == "64"
    assert command[command.index("--seed") + 1] == "0"


def test_server_values_remain_single_arguments():
    model = "model; touch /tmp/not-executed"
    command = build_server_command(ServerConfig(model=model), validate_gpus=False)
    assert command[2] == model
    assert len([part for part in command if "touch" in part]) == 1
