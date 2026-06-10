import modal
import subprocess
import os

# 1. Define the App and connect your volume
app = modal.App("asset-downloader")
vol = modal.Volume.from_name("denseclip-data")

# 2. Build a tiny environment just for downloading
image = modal.Image.debian_slim().apt_install("wget", "unzip")

@app.function(
    image=image,
    volumes={"/data": vol}, # Connect your volume to /data
    timeout=7200 # Give it up to 2 hours just in case
)
def download_files():
    print("🚀 Connecting to Cloud Volume...")
    os.chdir("/data")
    
    # --- 1. DOWNLOAD WEIGHTS ---
    print("🧠 Downloading OpenAI ResNet-50 Weights...")
    os.makedirs("/data/weights", exist_ok=True)
    subprocess.run([
        "wget", "-nc", "-O", "/data/weights/RN50.pt", 
        "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"
    ], check=True)
    
    # --- 2. DOWNLOAD DATASET ---
    print("🖼️ Downloading ADE20K Dataset from MIT (900MB+)...")
    subprocess.run([
        "wget", "-nc", 
        "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    ], check=True)
    
    print("📦 Unzipping ADE20K Dataset...")
    subprocess.run(["unzip", "-n", "-q", "ADEChallengeData2016.zip"], check=True)
    
    print("🧹 Cleaning up zip file to save volume space...")
    os.remove("ADEChallengeData2016.zip")
    
    # --- 3. VERIFY ---
    print("✅ All assets securely downloaded directly to the permanent volume!")
    subprocess.run(["ls", "-lah", "/data/weights"])
    subprocess.run(["ls", "-lah", "/data"])

@app.local_entrypoint()
def main():
    print("Deploying cloud-to-cloud download job...")
    download_files.remote()