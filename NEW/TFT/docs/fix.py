import subprocess
import sys

print("Fixing pytorch-forecasting compatibility issue...")
print("")

commands = [
    ["pip", "uninstall", "-y", "pytorch-lightning"],
    ["pip", "uninstall", "-y", "lightning"],
    ["pip", "install", "pytorch-lightning==1.9.5"],
    ["pip", "install", "--upgrade", "pytorch-forecasting"],
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print("Success")
    print("")

print("Done. Now run: python tft_training_final.py")