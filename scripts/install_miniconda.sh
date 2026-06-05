#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${1:-$HOME/miniconda3}"
ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64) MINICONDA_ARCH="x86_64" ;;
  aarch64|arm64) MINICONDA_ARCH="aarch64" ;;
  *) echo "Unsupported architecture: ${ARCH}" >&2; exit 1 ;;
esac

URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh"
INSTALLER="/tmp/miniconda-${MINICONDA_ARCH}.sh"

if command -v conda >/dev/null 2>&1; then
  echo "conda already exists: $(command -v conda)"
  conda --version
  exit 0
fi

if [ -d "${INSTALL_DIR}" ]; then
  echo "Install dir already exists but conda is not on PATH: ${INSTALL_DIR}" >&2
  echo "Activate it with: source ${INSTALL_DIR}/etc/profile.d/conda.sh" >&2
  exit 0
fi

echo "Downloading Miniconda from ${URL}"
if command -v curl >/dev/null 2>&1; then
  curl -L "${URL}" -o "${INSTALLER}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${INSTALLER}" "${URL}"
else
  echo "Need curl or wget to download Miniconda." >&2
  exit 1
fi

bash "${INSTALLER}" -b -p "${INSTALL_DIR}"
rm -f "${INSTALLER}"

echo "Miniconda installed at ${INSTALL_DIR}"
echo "Run the following before creating environments:"
echo "  source ${INSTALL_DIR}/etc/profile.d/conda.sh"
echo "  conda activate base"
