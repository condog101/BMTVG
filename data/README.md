# BMTVG data

This folder contains the segmentation point clouds (`seg1.ply`, `seg2.ply`,
`seg3.ply`), which live directly in this repository.

The large segmentation **videos** (`seg1.mkv`, `seg2.mkv`, `seg3.mkv`, ~31 GB
total) are hosted on the Hugging Face Hub to keep this Git repository small:

**https://huggingface.co/datasets/zcbecda/BMTVG**

## Downloading the videos

```bash
pip install huggingface_hub
huggingface-cli download zcbecda/BMTVG --repo-type dataset \
    --include "data/*.mkv" --local-dir .
```

This places `seg1.mkv`, `seg2.mkv`, `seg3.mkv` back into this `data/` folder.
