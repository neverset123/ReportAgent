#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
slides_dir="$repo_root/slides"
output_dir="$repo_root/frontend/pdf"

mkdir -p "$output_dir"

cd "$slides_dir"

shopt -s nullglob
files=(md/*.md)

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No markdown files found in slides/md."
  exit 0
fi

for file in "${files[@]}"; do
  echo "Processing $file"
  filename="$(basename -- "$file")"
  filename_without_extension="${filename%.*}"

  npm run export "$file"

  exported_file="./${filename_without_extension}-export.pdf"
  destination_file="$output_dir/${filename_without_extension}.pdf"

  if [[ -f "$exported_file" ]]; then
    mv "$exported_file" "$destination_file"
  else
    echo "File $exported_file does not exist, skipping move."
  fi
done

ls "$output_dir"
