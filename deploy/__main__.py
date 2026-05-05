"""Pulumi program — deploy one vLLM inference server per model on dedicated EC2 instances."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import pulumi
import pulumi_aws as aws

# ── config ────────────────────────────────────────────────────────────────────

config = pulumi.Config()
default_instance_type  = config.get("instance_type") or "g4dn.xlarge"
ssh_pub_key_path       = config.get("ssh_public_key_path") or "~/.ssh/id_ed25519.pub"
allowed_ssh_cidrs      = config.require_object("allowed_ssh_cidrs")
allowed_inference_cidrs = config.get_object("allowed_inference_cidrs") or ["0.0.0.0/0"]
hf_token               = config.require_secret("hf_token")
train_time             = config.get_int("train_time") or 300
max_experiments        = config.get_int("max_experiments") or 50
source_repo            = config.get("source_repo") or ""

INFERENCE_PORT = 8000


@dataclass
class ModelSpec:
    name: str
    model_id: str
    instance_type: str | None = None  # falls back to config instance_type
    gpu_memory_utilization: float = 0.90


_raw_models = config.get_object("models") or [
    {"name": "llama3",  "model_id": "meta-llama/Meta-Llama-3-8B-Instruct"},
    {"name": "mistral", "model_id": "mistralai/Mistral-7B-Instruct-v0.2"},
    {"name": "qwen",    "model_id": "Qwen/Qwen2.5-7B-Instruct"},
]
MODELS = [ModelSpec(**m) for m in _raw_models]

# ── AMI (Ubuntu 22.04 — standard, not Deep Learning AMI; user data installs drivers) ──

ami = aws.ec2.get_ami(
    most_recent=True,
    owners=["099720109477"],  # Canonical
    filters=[
        aws.ec2.GetAmiFilterArgs(
            name="name",
            values=["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"],
        ),
        aws.ec2.GetAmiFilterArgs(name="virtualization-type", values=["hvm"]),
        aws.ec2.GetAmiFilterArgs(name="architecture",        values=["x86_64"]),
    ],
)

# ── key pair ──────────────────────────────────────────────────────────────────

ssh_pub_key = Path(ssh_pub_key_path).expanduser().read_text().strip()
key_pair = aws.ec2.KeyPair("mat-model-key", public_key=ssh_pub_key)

# ── security group (shared across all model instances) ────────────────────────

sg = aws.ec2.SecurityGroup(
    "mat-model-sg",
    description="SSH + vLLM inference port for model servers",
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp", from_port=22, to_port=22,
            cidr_blocks=allowed_ssh_cidrs,
            description="SSH",
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp", from_port=INFERENCE_PORT, to_port=INFERENCE_PORT,
            cidr_blocks=allowed_inference_cidrs,
            description="vLLM inference",
        ),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            protocol="-1", from_port=0, to_port=0,
            cidr_blocks=["0.0.0.0/0"],
            description="All outbound",
        ),
    ],
)

# ── IAM role + instance profile (SSM for debug access) ───────────────────────

role = aws.iam.Role(
    "mat-model-role",
    assume_role_policy="""{
      "Version": "2012-10-17",
      "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]
    }""",
    tags={"Project": "MultiAgentTrainer"},
)
aws.iam.RolePolicyAttachment(
    "mat-model-ssm",
    role=role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore",
)
instance_profile = aws.iam.InstanceProfile("mat-model-profile", role=role.name)

# ── user data (templated per model) ───────────────────────────────────────────

_deploy_dir = Path(__file__).parent
_user_data_template   = (_deploy_dir / "user_data.sh").read_text()
_dockerfile_trainer   = (_deploy_dir / "Dockerfile.trainer").read_text()
_train_sh             = (_deploy_dir / "train.sh").read_text()
_mat_query            = (_deploy_dir / "mat-query").read_text()


def _user_data(model: ModelSpec, token: str) -> str:
    return _user_data_template.format(
        hf_token=token,
        model_id=model.model_id,
        port=INFERENCE_PORT,
        gpu_memory_utilization=model.gpu_memory_utilization,
        train_time=train_time,
        max_experiments=max_experiments,
        source_repo=source_repo,
        dockerfile_trainer=_dockerfile_trainer,
        train_sh=_train_sh,
        mat_query=_mat_query,
    )


# ── one instance per model ────────────────────────────────────────────────────

instances: list[aws.ec2.Instance] = []

for model in MODELS:
    inst = hf_token.apply(
        lambda token, m=model: aws.ec2.Instance(
            f"mat-model-{m.name}",
            ami=ami.id,
            instance_type=m.instance_type or default_instance_type,
            key_name=key_pair.key_name,
            vpc_security_group_ids=[sg.id],
            iam_instance_profile=instance_profile.name,
            user_data=_user_data(m, token),
            root_block_device=aws.ec2.InstanceRootBlockDeviceArgs(
                volume_size=100,
                volume_type="gp3",
                delete_on_termination=True,
            ),
            tags={
                "Name": f"mat-model-{m.name}",
                "Project": "MultiAgentTrainer",
                "Model": m.model_id,
            },
        )
    )
    instances.append(inst)

# ── outputs ───────────────────────────────────────────────────────────────────

pulumi.export("instance_ids", [inst.apply(lambda i: i.id) for inst in instances])
pulumi.export("public_ips",   [inst.apply(lambda i: i.public_ip) for inst in instances])

pulumi.export(
    "model_endpoints",
    pulumi.Output.all(*[inst.apply(lambda i: i.public_ip) for inst in instances]).apply(
        lambda ips: {m.name: f"http://{ip}:{INFERENCE_PORT}/v1" for m, ip in zip(MODELS, ips)}
    ),
)

pulumi.export(
    "agenttester_yaml_snippet",
    pulumi.Output.all(*[inst.apply(lambda i: i.public_ip) for inst in instances]).apply(
        lambda ips: "\n".join(
            textwrap.dedent(f"""\
                  {m.name}:
                    command: 'python scripts/query_model.py http://{ip}:{INFERENCE_PORT} {m.model_id} {{prompt}}'
                    host: localhost
                    commit_style: manual
                    timeout: 120""")
            for m, ip in zip(MODELS, ips)
        )
    ),
)
