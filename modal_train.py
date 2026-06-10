import modal

# App
app = modal.App("denseclip-thesis-training")

# Image
image = (
    modal.Image.debian_slim()
    .add_local_dir(".", remote_path="/workspace", copy=True)
    .pip_install("opencv-python-headless", "wandb", "timm")
    .pip_install_from_requirements("requirements.txt")
)

# Persistent volume
vol = modal.Volume.from_name(
    "denseclip-data",
    create_if_missing=True
)

@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=16384,
    timeout=86400,  # 24 hours
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_training():
    import os
    import subprocess

    print("🚀 Booting L4 instance...")
    print("📂 Dataset path: /data/ADEChallengeData2016")
    print("📂 Checkpoints path: /data/checkpoints")
    print("📂 Weights path: /data/weights")

    # Move into copied project directory
    os.chdir("/workspace")

    subprocess.run(
        ["python", "train_segmentation.py"],
        check=True
    )

@app.local_entrypoint()
def main():
    print("🚀 Launching DenseCLIP training on Modal...")
    run_training.remote()