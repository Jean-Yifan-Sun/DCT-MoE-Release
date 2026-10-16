#!/usr/bin/env bash
# Portable implementation of the EchoNet-Synthetic frame extraction command.
# Same FFmpeg select/format filter, JPG naming, and default 32-way parallelism.
# Reference: https://github.com/HReynaud/EchoNet-Synthetic/blob/87189437cce990daf195f54121f5d4707a5f3be6/scripts/extract_frames_from_videos.sh
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <path_to_videos_folder> <output_folder>" >&2
    exit 1
fi
command -v ffmpeg >/dev/null || { echo 'ffmpeg must be installed and on PATH' >&2; exit 1; }
[[ -d "$1" ]] || { echo "Video folder does not exist: $1" >&2; exit 1; }
input_folder="$1"
output_folder="$2"
workers="${WORKERS:-32}"
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { echo 'WORKERS must be a positive integer' >&2; exit 1; }
# A flat output directory requires unique video basenames.
[[ ! -e "$output_folder" || -d "$output_folder" ]] || exit 1
mkdir -p "$output_folder"
if [[ -n "$(find "$output_folder" -mindepth 1 -print -quit)" ]]; then
    echo 'Output folder must be empty to avoid stale/overwritten frames' >&2
    exit 1
fi
export output_folder
extract_frames() {
    set -euo pipefail
    local video_file="$1" video_name base_name
    video_name="$(basename "$video_file")"
    base_name="${video_name%.*}"
    ffmpeg -nostdin -loglevel error -n -i "$video_file" \
        -vf 'select=not(mod(n\,5)),format=gray,format=yuv420p' \
        -vsync vfr "$output_folder/${base_name}_%05d.jpg"
}
export -f extract_frames
find "$input_folder" -name '*.avi' -print0 | xargs -0 -I {} -P "$workers" bash -c 'extract_frames "$1"' _ '{}'
