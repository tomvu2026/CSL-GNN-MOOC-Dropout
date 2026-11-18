#!/bin/bash
# Download the two public datasets (AMiner MOOCData) and convert them to the common parquet schema.
set -e; source "$(dirname "$0")/common.sh"
D=${CSL_DATA:-data}; mkdir -p "$D/kddcup15/raw" "$D/xuetangx/raw"
[ -f "$D/kddcup15/raw/kddcup15.zip" ] || wget -O "$D/kddcup15/raw/kddcup15.zip" https://lfs.aminer.cn/misc/moocdata/data/kddcup15.zip
[ -d "$D/kddcup15/raw/kddcup15" ] || unzip -q "$D/kddcup15/raw/kddcup15.zip" -d "$D/kddcup15/raw"
[ -f "$D/xuetangx/raw/prediction_data.tar.gz" ] || wget -O "$D/xuetangx/raw/prediction_data.tar.gz" https://lfs.aminer.cn/misc/moocdata/data/prediction_data.tar.gz
[ -d "$D/xuetangx/raw/prediction_log" ] || tar -xzf "$D/xuetangx/raw/prediction_data.tar.gz" -C "$D/xuetangx/raw"
python3 src/prep_kdd.py
python3 src/prep_xuetangx.py
