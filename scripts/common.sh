# sourced by the run scripts: repository root, logs folder, shared CSL-GNN options (selected configuration, Table 3)
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGS=${CSL_RESULTS:-results}/logs; mkdir -p "$LOGS"
CSL_OPTS="--attn 1 --temporal 1 --jk 1 --wd 0.0001 --fullfit 1 --out fullfit"
