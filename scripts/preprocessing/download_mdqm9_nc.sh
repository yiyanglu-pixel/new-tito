#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-/localhome3/lyy/repro/tito/datasets/mdqm9-nc}"
BASE_URL="https://zenodo.org/records/10579242/files"
JOBS="${JOBS:-4}"

mkdir -p "${DATA_DIR}/parts" "${DATA_DIR}/splits"

download_and_check() {
  local name="$1"
  local output="$2"
  local expected_md5="$3"

  if [[ -f "${output}" ]] && echo "${expected_md5}  ${output}" | md5sum -c --status; then
    echo "OK ${output}"
    return 0
  fi

  echo "Downloading ${name} -> ${output}"
  wget -c --tries=0 --retry-connrefused --timeout=60 --waitretry=10 \
    -O "${output}" "${BASE_URL}/${name}?download=1"

  echo "${expected_md5}  ${output}" | md5sum -c
}

download_splits() {
  local source_dir="${DATA_DIR}/../_sources/mdqm9-nc-loaders-main"
  if [[ ! -f "${DATA_DIR}/splits/train_indices.npy" ]]; then
    mkdir -p "${DATA_DIR}/../_sources"
    (
      cd "${DATA_DIR}/../_sources"
      rm -rf mdqm9-nc-loaders-main mdqm9-nc-loaders-main.zip
      wget -q -O mdqm9-nc-loaders-main.zip \
        https://codeload.github.com/olsson-group/mdqm9-nc-loaders/zip/refs/heads/main
      unzip -q mdqm9-nc-loaders-main.zip
    )
    cp -a "${source_dir}/splits/." "${DATA_DIR}/splits/"
  fi
}

download_splits

download_and_check "mdqm9-nc.sdf" "${DATA_DIR}/mdqm9-nc.sdf" "f65e993083ce1209594fc5a978523b63"

part_specs=(
  "mdqm9-nc_00 7a12b671d50ea368db29ce56818909aa"
  "mdqm9-nc_01 c35ca27ac6ba5bbf7dc7043271cff7f8"
  "mdqm9-nc_02 eee13f065065077560a01b3fc8e5f052"
  "mdqm9-nc_03 b4959de2e7de83326e5986533ba5d9d9"
  "mdqm9-nc_04 f2cdc2ff340ebfd818c447e604ffe2e2"
  "mdqm9-nc_05 1b078ef0c22f722af5fc7f2ead4db78c"
  "mdqm9-nc_06 899cd098c63f77b92cc9a3f5814f7686"
  "mdqm9-nc_07 400949066ee1a155b5a538d4290687c7"
  "mdqm9-nc_08 efade1b03383f5e6d19ef6aab480bdcd"
  "mdqm9-nc_09 32564278660788124d1704b2babbc813"
)

echo "Downloading HDF5 shards with up to ${JOBS} parallel jobs"
status=0
for spec in "${part_specs[@]}"; do
  read -r name md5 <<< "${spec}"
  download_and_check "${name}" "${DATA_DIR}/parts/${name}" "${md5}" &

  while [[ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]]; do
    wait -n || status=1
  done
done

while [[ "$(jobs -rp | wc -l)" -gt 0 ]]; do
  wait -n || status=1
done

if [[ "${status}" -ne 0 ]]; then
  echo "At least one shard failed to download or verify." >&2
  exit "${status}"
fi

if [[ ! -f "${DATA_DIR}/mdqm9-nc.hdf5" ]]; then
  echo "Merging HDF5 shards into ${DATA_DIR}/mdqm9-nc.hdf5"
  cat "${DATA_DIR}"/parts/mdqm9-nc_* > "${DATA_DIR}/mdqm9-nc.hdf5"
fi

echo "MDQM9-nc is ready at ${DATA_DIR}"
