"""SSH-based TPU kernel evaluation. Runs evaluate.py on a remote TPU-VM over SSH.

Drop-in alternative to KubeEvaluator for TPU-VM setups (e.g. a v6e-16 slice)
that have no GKE cluster. Instead of a K8s Job + ConfigMap + GCS bucket, this:

  1. writes the base64 payload to a remote temp file (piped over ssh),
  2. runs `docker/evaluate.py` on the VM with the v6e roofline env + IR dump
     flags (evaluate.py sets its own dump flags and preserves existing env),
  3. parses `EVAL_RESULT:` lines from stdout (same wire format as KubeEvaluator),
  4. scp's artifacts (llo_final.txt / hlo_post_opt.txt / trace_events.json) back
     from ARTIFACTS_LOCAL_DIR on the VM — no GCS involved.

The remote VM must have the repo checked out at `remote_repo_dir` and the
kernel-evolve package installed into `python_bin`'s env
(`uv pip install -e kernel-evolve/` or equivalent), plus a working TPU runtime.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from kernel_evolve.evaluator import (
  BatchEvalRequest,
  BatchEvalResult,
  EvalRequest,
  EvalResult,
  Evaluator,
)


@dataclass
class SSHConfig:
  host: str
  user: str = "cloud-user"
  ssh_key: str = ""
  # Where the repo is checked out on the VM (contains kernel-evolve/).
  remote_repo_dir: str = "/home/cloud-user/Glaucis"
  # Python that has kernel-evolve + jax/libtpu installed.
  python_bin: str = "/home/cloud-user/tpu-venv/bin/python"
  remote_tmp: str = "/tmp/kernel_eval"
  remote_artifacts_dir: str = "/tmp/kernel_eval_artifacts"
  # Where scp'd artifacts land locally (per variant under artifacts/<variant_id>/).
  local_artifacts_dir: str = "artifacts"
  timeout: int = 1800
  # v6e (Trillium) roofline — passed through as env to evaluate.py.
  peak_flops: float = 918e12
  peak_hbm_bw: float = 1759e9
  hbm_capacity_gb: float = 32.0
  vmem_capacity_mib: float = 128.0
  # Extra libtpu flags appended (evaluate.py already sets the dump flags).
  extra_libtpu_init_args: str = ""
  cleanup: bool = True
  ssh_opts: tuple[str, ...] = (
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "BatchMode=yes",
    "-o", "ServerAliveInterval=30",
  )


class SSHEvaluator(Evaluator):
  """Evaluates kernel variants by running evaluate.py on a remote TPU-VM via SSH."""

  def __init__(self, config: SSHConfig):
    self._config = config

  # ── low-level ssh/scp plumbing ──────────────────────────────────────────

  def _key_args(self) -> list[str]:
    return ["-i", self._config.ssh_key] if self._config.ssh_key else []

  def _target(self) -> str:
    return f"{self._config.user}@{self._config.host}"

  async def _run(self, *args: str, stdin: str | None = None, timeout: int | None = None) -> tuple[str, str, int]:
    proc = await asyncio.create_subprocess_exec(
      *args,
      stdin=asyncio.subprocess.PIPE if stdin is not None else None,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )
    try:
      stdout, stderr = await asyncio.wait_for(
        proc.communicate(input=stdin.encode() if stdin is not None else None),
        timeout=timeout,
      )
    except asyncio.TimeoutError:
      proc.kill()
      return "", f"command timed out after {timeout}s", 124
    return stdout.decode(errors="replace"), stderr.decode(errors="replace"), proc.returncode

  async def _ssh(self, remote_cmd: str, stdin: str | None = None, timeout: int | None = None) -> tuple[str, str, int]:
    args = ["ssh", *self._key_args(), *self._config.ssh_opts, self._target(), remote_cmd]
    return await self._run(*args, stdin=stdin, timeout=timeout)

  async def _scp_from(self, remote_path: str, local_path: str) -> tuple[str, str, int]:
    args = ["scp", "-r", *self._key_args(), *self._config.ssh_opts,
            f"{self._target()}:{remote_path}", local_path]
    return await self._run(*args, timeout=300)

  # ── env / command construction ──────────────────────────────────────────

  def _env_prefix(self, artifacts_dir: str) -> str:
    c = self._config
    env = {
      "JAX_PLATFORMS": "tpu,cpu",
      "ENABLE_PJRT_COMPATIBILITY": "true",
      "PEAK_FLOPS": repr(c.peak_flops),
      "PEAK_HBM_BW": repr(c.peak_hbm_bw),
      "HBM_CAPACITY_GB": repr(c.hbm_capacity_gb),
      "VMEM_CAPACITY_MIB": repr(c.vmem_capacity_mib),
      "ARTIFACTS_LOCAL_DIR": artifacts_dir,
    }
    if c.extra_libtpu_init_args:
      env["LIBTPU_INIT_ARGS"] = c.extra_libtpu_init_args
    return " ".join(f"{k}={v}" for k, v in env.items())

  def _eval_cmd(self, payload_file: str, artifacts_dir: str) -> str:
    c = self._config
    return (
      f"cd {c.remote_repo_dir}/kernel-evolve && "
      f"{self._env_prefix(artifacts_dir)} "
      f"{c.python_bin} docker/evaluate.py --eval-payload-file {payload_file}"
    )

  @staticmethod
  def _parse_results(logs: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in logs.splitlines():
      if "EVAL_RESULT:" in line:
        raw = line.split("EVAL_RESULT:", 1)[1].strip()
        try:
          data = json.loads(raw)
        except json.JSONDecodeError:
          print(f"Malformed EVAL_RESULT: {raw[:200]}", file=sys.stderr)
          continue
        out[data.get("variant_id", "unknown")] = data
    return out

  async def _pull_artifacts(self, result: dict) -> None:
    remote_dir = (result.get("metadata") or {}).get("artifacts_local_dir")
    if not remote_dir:
      return
    vid = result.get("variant_id", "unknown")
    local = Path(self._config.local_artifacts_dir) / vid
    local.mkdir(parents=True, exist_ok=True)
    # copy the *contents* of the remote variant dir into local/<vid>/
    _, stderr, rc = await self._scp_from(f"{remote_dir}/.", str(local))
    if rc != 0:
      print(f"Artifact scp for {vid} failed: {stderr.strip()}", file=sys.stderr)

  async def _write_payload(self, payload: str, remote_file: str) -> int:
    _, stderr, rc = await self._ssh(
      f"mkdir -p {self._config.remote_tmp} && cat > {remote_file}",
      stdin=payload,
      timeout=120,
    )
    if rc != 0:
      print(f"Failed to write remote payload: {stderr.strip()}", file=sys.stderr)
    return rc

  async def _cleanup(self, *remote_paths: str) -> None:
    if not self._config.cleanup or not remote_paths:
      return
    await self._ssh("rm -rf " + " ".join(remote_paths), timeout=60)

  # ── public API (parallels KubeEvaluator) ────────────────────────────────

  async def evaluate(self, request: EvalRequest) -> EvalResult:
    job = re.sub(r"[^a-z0-9_.-]", "-", request.variant_id.lower()) or "variant"
    payload_file = f"{self._config.remote_tmp}/{job}.payload"
    artifacts_dir = f"{self._config.remote_artifacts_dir}/{job}"

    if await self._write_payload(request.encode_b64(), payload_file) != 0:
      return EvalResult.compile_error("Failed to stage payload on remote VM")

    stdout, stderr, rc = await self._ssh(
      self._eval_cmd(payload_file, artifacts_dir), timeout=self._config.timeout
    )
    results = self._parse_results(stdout)
    result = results.get(request.variant_id)

    if result is None:
      await self._cleanup(payload_file, artifacts_dir)
      tail = (stdout + stderr)[-500:]
      return EvalResult.compile_error(f"No EVAL_RESULT from remote. Tail: {tail}")

    await self._pull_artifacts(result)
    await self._cleanup(payload_file, artifacts_dir)
    return EvalResult.from_dict(result)

  async def evaluate_batch(self, batch_request: BatchEvalRequest) -> BatchEvalResult:
    n = len(batch_request.variants)
    job = f"batch-{n}v"
    payload_file = f"{self._config.remote_tmp}/{job}.payload"
    artifacts_dir = self._config.remote_artifacts_dir

    if await self._write_payload(batch_request.encode_b64(), payload_file) != 0:
      err = EvalResult.compile_error("Failed to stage batch payload on remote VM")
      return BatchEvalResult(results={v["variant_id"]: err for v in batch_request.variants})

    timeout = 300 * n + 300
    stdout, stderr, rc = await self._ssh(
      self._eval_cmd(payload_file, artifacts_dir), timeout=max(timeout, self._config.timeout)
    )
    parsed = self._parse_results(stdout)

    results: dict[str, EvalResult] = {}
    for v in batch_request.variants:
      vid = v["variant_id"]
      data = parsed.get(vid)
      if data is None:
        results[vid] = EvalResult.compile_error("No EVAL_RESULT found in remote logs")
        continue
      await self._pull_artifacts(data)
      results[vid] = EvalResult.from_dict(data)

    await self._cleanup(payload_file, artifacts_dir)
    return BatchEvalResult(results=results)


def ssh_config_from_evaluator(evaluator: dict, roofline: dict | None = None) -> SSHConfig:
  """Build an SSHConfig from parsed config dicts (evaluator + optional roofline)."""
  r = roofline or {}
  return SSHConfig(
    host=evaluator["ssh_host"],
    user=evaluator.get("ssh_user", "cloud-user"),
    ssh_key=evaluator.get("ssh_key", ""),
    remote_repo_dir=evaluator.get("ssh_remote_repo", "/home/cloud-user/Glaucis"),
    python_bin=evaluator.get("ssh_python", "/home/cloud-user/tpu-venv/bin/python"),
    remote_tmp=evaluator.get("ssh_remote_tmp", "/tmp/kernel_eval"),
    remote_artifacts_dir=evaluator.get("ssh_artifacts_dir", "/tmp/kernel_eval_artifacts"),
    timeout=evaluator.get("timeout", 1800),
    peak_flops=float(r.get("peak_flops", 918e12)),
    peak_hbm_bw=float(r.get("peak_hbm_bw", 1759e9)),
    hbm_capacity_gb=float(r.get("hbm_capacity_gb", 32.0)),
    vmem_capacity_mib=float(r.get("vmem_capacity_mib", 128.0)),
    extra_libtpu_init_args=evaluator.get("ssh_extra_libtpu_init_args", ""),
  )
