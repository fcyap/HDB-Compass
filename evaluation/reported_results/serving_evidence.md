# Gemma serving evidence

Selected lines from the saved vLLM server-log excerpts, collected 11 Nov 2026. Local paths, account names,
ports and startup command lines are omitted. These are transcribed measurements, not a new server run.

- Job 363139: first attempt, 24 GB slice, requested 32,768; startup failed with an estimated 15,488-token maximum.
- Job 363193: second attempt, L40S at 65,536; server started, application smoke test failed.
- Job 363197: final benchmark attempt, L40S at 65,536; smoke test deliberately skipped.

The original startup error states that a 32,768 sequence needs 1.97 GiB of KV cache, versus 1.63 GiB available.
The final server's total cache capacity is not its per-request context limit: 1,199,893 cache tokens were available,
with a configured 65,536-token maximum per request. The benchmark sent requests sequentially.

```text
## job 363139
NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 610.57.04, 97887 MiB
(EngineCore pid=292874) INFO 11-11 18:52:36 [model_runner.py:428] Model loading took 18.13 GiB memory and 11.854233 seconds
(EngineCore pid=292874) INFO 11-11 18:53:32 [gpu_worker.py:772] Available KV cache memory: 1.63 GiB
(EngineCore pid=292874) INFO 11-11 18:53:32 [gpu_worker.py:787] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.9500 is equivalent to --gpu-memory-utilization=0.9275 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9725. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
## job 363193
NVIDIA L40S, 590.48.01, 46068 MiB
(EngineCore pid=4018396) INFO 11-11 19:53:53 [model_runner.py:428] Model loading took 18.12 GiB memory and 18.000685 seconds
(EngineCore pid=4018396) INFO 11-11 19:54:56 [gpu_worker.py:772] Available KV cache memory: 21.98 GiB
(EngineCore pid=4018396) INFO 11-11 19:54:56 [gpu_worker.py:787] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.9200 is equivalent to --gpu-memory-utilization=0.9173 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9227. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(EngineCore pid=4018396) INFO 11-11 19:54:57 [kv_cache_utils.py:1147] GPU KV cache size: 1,199,893 tokens, Maximum concurrency for 65,536 tokens per request: 18.31x
## job 363197
NVIDIA L40S, 590.48.01, 46068 MiB
(EngineCore pid=4103436) INFO 11-11 20:03:13 [model_runner.py:428] Model loading took 18.12 GiB memory and 13.020292 seconds
(EngineCore pid=4103436) INFO 11-11 20:03:37 [gpu_worker.py:772] Available KV cache memory: 21.98 GiB
(EngineCore pid=4103436) INFO 11-11 20:03:37 [gpu_worker.py:787] CUDA graph memory profiling is enabled (default since v0.21.0). The current --gpu-memory-utilization=0.9200 is equivalent to --gpu-memory-utilization=0.9173 without CUDA graph memory profiling. To maintain the same effective KV cache size as before, increase --gpu-memory-utilization to 0.9227. To disable, set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
(EngineCore pid=4103436) INFO 11-11 20:03:37 [kv_cache_utils.py:1147] GPU KV cache size: 1,199,893 tokens, Maximum concurrency for 65,536 tokens per request: 18.31x
```
