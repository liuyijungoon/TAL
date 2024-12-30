import os
import sys
import wandb
import subprocess

def main():
    # 初始化wandb
    wandb.init()

    # 获取当前运行的配置
    config = wandb.config

    # 从config中获取nproc_per_node的值
    nproc_per_node = config.get('nproc_per_node', 1)  # 默认值为1

    # 构建命令行参数
    cmd = [
        "python", "-m", "torch.distributed.launch",
        f"--nproc_per_node={nproc_per_node}",
        "main_tal_mgpu.py"
    ]

    # 添加wandb配置作为命令行参数
    for key, value in config.items():
        if key != 'nproc_per_node':  # 排除nproc_per_node，因为它已经被用于torch.distributed.launch
            cmd.append(f"--{key}={value}")

    # 打印命令以便调试
    print(f"Running command: {' '.join(cmd)}")

    # 运行命令
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()