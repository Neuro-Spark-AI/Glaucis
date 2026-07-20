"""YAML config parsing and validation with Pydantic."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class EvolveMarkers(BaseModel):
  start: str = "# EVOLVE-BLOCK-START"
  end: str = "# EVOLVE-BLOCK-END"


class KernelConfig(BaseModel):
  name: str
  template: str
  reference: str
  evolve_markers: EvolveMarkers = Field(default_factory=EvolveMarkers)


class CorrectnessConfig(BaseModel):
  method: str = "allclose"
  rtol: float = 1e-2
  atol: float = 1e-2


class EvaluatorConfig(BaseModel):
  # "kube" (GKE K8s Job) or "ssh" (run evaluate.py on a TPU-VM over SSH).
  type: str = "kube"

  # --- kube transport ---
  namespace: str = "default"
  job_template: str = ".github/ci/kernel-eval-job.yaml"
  repo: str = ""
  branch: str = "main"
  poll_interval: int = 15
  timeout: int = 600

  # --- ssh transport (type: ssh) ---
  ssh_host: str = ""
  # For a multi-host ICI slice (e.g. v6e-16 = 4 hosts), list EVERY worker's IP.
  # evaluate.py must be co-launched on all of them at once: the hosts share one
  # ICI domain and libtpu blocks on a slice-wide init barrier, so a single-host
  # invocation hangs forever waiting for the peers. Results and artifacts are
  # collected from the first host in the list (the "primary"); the others run
  # the same program only to satisfy the barrier. When empty, falls back to the
  # single-host [ssh_host] (correct for single-host slices: v6e-1/-4/-8).
  ssh_hosts: list[str] = Field(default_factory=list)
  ssh_user: str = "cloud-user"
  ssh_key: str = ""
  ssh_remote_repo: str = "/home/cloud-user/Glaucis"
  ssh_python: str = "/home/cloud-user/tpu-venv/bin/python"
  ssh_remote_tmp: str = "/tmp/kernel_eval"
  ssh_artifacts_dir: str = "/tmp/kernel_eval_artifacts"
  ssh_extra_libtpu_init_args: str = ""

  def resolved_ssh_hosts(self) -> list[str]:
    """All worker hosts to co-launch on; first is the primary. Falls back to [ssh_host]."""
    hosts = [h for h in self.ssh_hosts if h]
    if hosts:
      return hosts
    return [self.ssh_host] if self.ssh_host else []


class RooflineConfig(BaseModel):
  """Per-chip peak numbers for roofline metrics. Defaults are TPU v6e (Trillium)."""
  peak_flops: float = 918e12        # bf16 TFLOP/s
  peak_hbm_bw: float = 1759e9       # HBM GB/s
  hbm_capacity_gb: float = 32.0     # HBM per chip
  vmem_capacity_mib: float = 128.0  # physical VMEM per chip


class TPUConfig(BaseModel):
  cluster: str = ""
  zone: str = ""


class SessionConfig(BaseModel):
  max_iterations: int = 20
  output_dir: str = "runs/default"
  # GitHub repo (owner/name) for the tracking Issue, reflect comments, and the
  # final PR. Empty = the current checkout's origin. Set this to a PRIVATE repo
  # when the kernels are private, so variant code / results / the optimized
  # kernel never post to a public fork (e.g. a public engine fork like Glaucis).
  tracking_repo: str = ""
  # Path to the cross-session learnings file. Default is the repo-root AGENT.md.
  # Point it at a private-repo checkout for private kernels so learnings don't
  # land in a public fork.
  agent_md_path: str = "AGENT.md"


class BatchConfig(BaseModel):
  variants_per_round: int = Field(default=1, ge=1)
  top_k: int = Field(default=1, ge=1)
  max_active_lineages: int = Field(default=4, ge=1)


class EvolveConfig(BaseModel):
  model_config = {"extra": "forbid"}

  kernel: KernelConfig
  shapes: list[dict[str, Any]]
  correctness: CorrectnessConfig = Field(default_factory=CorrectnessConfig)
  evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
  roofline: RooflineConfig = Field(default_factory=RooflineConfig)
  tpu: TPUConfig = Field(default_factory=TPUConfig)
  session: SessionConfig = Field(default_factory=SessionConfig)
  batch: BatchConfig = Field(default_factory=BatchConfig)


def load_config(path: str | Path) -> EvolveConfig:
  """Load and validate an EvolveConfig from a YAML file."""
  with open(path) as f:
    data = yaml.safe_load(f)
  return EvolveConfig(**data)
