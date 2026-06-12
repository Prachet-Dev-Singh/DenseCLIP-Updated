import modal
import subprocess
import os

# 1. Define the App
app = modal.App("denseclip-thesis-eval")

# Persistent volume
vol = modal.Volume.from_name("denseclip-data", create_if_missing=True)

# 2. Build the environment (Using YOUR .add_local_dir fix!)
image = (
    modal.Image.debian_slim()
    .add_local_dir(".", remote_path="/workspace", copy=True)
    .pip_install("opencv-python-headless", "wandb", "timm", "matplotlib")
    .pip_install_from_requirements("requirements.txt")
)

# 3. Define the cloud function
@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=16384,
    timeout=3600, # 1 hour is plenty for an evaluation pass
    volumes={"/data": vol},
    # Notice we completely removed the buggy `mounts=` argument here!
)
def run_eval():
    print("🚀 Booting L4 Evaluation Engine...")
    
    # Move into the copied project directory
    os.chdir("/workspace")
    
    # Execute your mathematically corrected eval script
    subprocess.run([
        'python', 'eval_segmentation.py',
        '--checkpoint', '/data/checkpoints/denseclip_step_80000.pth',
        '--dataset',    '/data/ADEChallengeData2016',
        '--output_dir', '/data/eval_results',
        '--num_vis',    '20',
    ], check=True)

# 4. The local execution trigger
@app.local_entrypoint()
def main():
    print("Deploying evaluation job to Modal...")
    run_eval.remote()