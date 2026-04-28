#!/usr/bin/env bash
# Download ISPD'15 + TILOS-AI MacroPlacement benchmarks + ASAP7 PDK.
# Run from project root: bash scripts/download_data.sh
set -e

DATA=$(dirname "$(realpath "$0")")/../data
mkdir -p "$DATA"

echo "[1/3] ISPD 2015 contest benchmarks..."
mkdir -p "$DATA/ispd15"
# Mirror at UCSD — exact URL TBD when user runs this
# wget -O "$DATA/ispd15.tar.gz" https://...ispd2015.tar.gz
# tar -xzf "$DATA/ispd15.tar.gz" -C "$DATA/ispd15"
echo "  TODO: replace with actual mirror URL"

echo "[2/3] TILOS-AI MacroPlacement (Ariane, MemPool, NVDLA, BlackParrot)..."
mkdir -p "$DATA/tilos"
# git clone https://github.com/TILOS-AI-Institute/MacroPlacement "$DATA/tilos"
echo "  TODO: git clone TILOS-AI MacroPlacement repo"

echo "[3/3] ASAP7 PDK + NanGate45 PDK..."
mkdir -p "$DATA/pdks/asap7"
mkdir -p "$DATA/pdks/nangate45"
# git clone https://github.com/The-OpenROAD-Project/asap7
# wget NanGate45 from OpenROAD-flow-scripts
echo "  TODO: clone OpenROAD-flow-scripts which includes both PDKs"

echo "Done."
