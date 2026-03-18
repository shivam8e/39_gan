import os
import json
import subprocess

# paths
DATASET_JSON = "styled_data/dataset.json"
NETWORK = "network-snapshot.pkl"   # update path if needed
OUTPUT_DIR = "generated"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# load labels
with open(DATASET_JSON, "r") as f:
    data = json.load(f)

labels = data["labels"]

# get unique class IDs
class_ids = sorted(list(set([label for _, label in labels])))

print(f"Found {len(class_ids)} persons")

# generate images for each class
for class_id in class_ids:
    outdir = os.path.join(OUTPUT_DIR, f"person_{class_id}")
    os.makedirs(outdir, exist_ok=True)

    print(f"Generating for person {class_id}...")

    cmd = f"""
    python generate.py \
    --outdir={outdir} \
    --seeds=1-30 \
    --class={class_id} \
    --network={NETWORK}
    """

    subprocess.run(cmd, shell=True)

print("Done generating all persons")
