#!/usr/bin/env bash
# Download the Mooncake FAST'25 traces at a pinned commit and verify their SHA-256.
set -euo pipefail

COMMIT=5872ee870dad0a65fe3557786f66cd7d13405816
BASE="https://raw.githubusercontent.com/kvcache-ai/Mooncake/${COMMIT}/FAST25-release/traces"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/traces"

declare -A SHA=(
  [conversation_trace.jsonl]=b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df
  [toolagent_trace.jsonl]=48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71
  [synthetic_trace.jsonl]=bd070915a98fc0ed264d7cfef2ce746002eb3076a695ec31ba2674c0111ec131
)

mkdir -p "$DEST"
for f in "${!SHA[@]}"; do
  if [[ ! -f "$DEST/$f" ]]; then
    curl -fsSL -o "$DEST/$f.part" "$BASE/$f"
    mv "$DEST/$f.part" "$DEST/$f"
  fi
  echo "${SHA[$f]}  $DEST/$f" | sha256sum -c --quiet
done
echo "traces OK in $DEST (Mooncake commit $COMMIT)"
