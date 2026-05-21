#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
MODES=(age gender ethnicity)

for mode in "${MODES[@]}"; do
  echo "[INFO] Running FireRed identity experiment: ${mode}"
  ID_EDIT_MODE="${mode}" "${PYTHON_BIN}" firered_edit_changeperson.py

  echo "[INFO] Running FLUX.2 Klein identity experiment: ${mode}"
  ID_EDIT_MODE="${mode}" "${PYTHON_BIN}" flux2_klein_edit_changeperson.py
done

echo "[INFO] All identity experiments finished."
