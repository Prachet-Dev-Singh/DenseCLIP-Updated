import modal

# 1. Define the App name
app = modal.App("denseclip-thesis-training")

# 2. Build the remote container environment using your requirements.txt
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("opencv-python-headless")  # Ensures seamless image processing on cloud servers
    .from_requirements("requirements.txt")    # Automatically installs your other local dependencies
)

# 3. Connect to your persistent cloud drive
vol = modal.Volume.from_name("denseclip-data")

# 4. Define the remote execution settings
@app.function(
    image=image,
    gpu="L4",               # Ada Lovelace architecture, 24GB VRAM ($0.80/hr)
    cpu=4.0,                # 4 CPU cores, perfectly matching your 4 dataloader workers
    memory=16384,           # 16 GB of system RAM to handle large image batches without bottlenecking
    timeout=86400,          # 24-hour limit
    volumes={"/data": vol}, 
    mounts=[modal.Mount.from_local_dir(".", remote_path="/workspace")] 
)
def run_training():
    import subprocess
    import os
    
    print("🚀 Booting A100 Instance and establishing workspace...")
    
    # Switch to the directory holding your mounted code files
    os.chdir("/workspace")
    
    # Execute the pure PyTorch training pipeline
    subprocess.run(["python", "train_segmentation.py"], check=True)

# 5. Local machine entry point
@app.local_entrypoint()
def main():
    print("Deploying training job to Modal Serverless GPU infrastructure...")
    run_training.remote()