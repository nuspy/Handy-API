#!/bin/bash
# Patches whisper-rs-sys 0.13.1 in the cargo registry for Windows/MSVC builds.
# Idempotent — safe to run multiple times.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCHES_DIR="$SCRIPT_DIR/patches"
REGISTRY_BASE="$HOME/.cargo/registry/src"
CRATE_DIR=$(find "$REGISTRY_BASE" -maxdepth 2 -type d -name "whisper-rs-sys-0.13.1" 2>/dev/null | head -1)

if [ -z "$CRATE_DIR" ]; then
    echo "[patch-whisper] whisper-rs-sys-0.13.1 not in registry — skipping"
    exit 0
fi

BUILD_RS="$CRATE_DIR/build.rs"
BINDINGS_RS="$CRATE_DIR/src/bindings.rs"

# ── 1) Patch build.rs ────────────────────────────────────────────────────────
if grep -q "PATCHED_FOR_WINDOWS" "$BUILD_RS" 2>/dev/null; then
    echo "[patch-whisper] build.rs already patched"
else
    echo "[patch-whisper] Patching build.rs ..."
    python3 -c "
import sys, os
build_rs = sys.argv[1]
old_file = sys.argv[2]
new_file = sys.argv[3]

with open(build_rs, 'r') as f:
    content = f.read()
with open(old_file, 'r') as f:
    old = f.read().rstrip('\n')
with open(new_file, 'r') as f:
    new = f.read().rstrip('\n')

if old not in content:
    print('[patch-whisper] ERROR: expected code not found in build.rs')
    sys.exit(1)

content = content.replace(old, new, 1)
with open(build_rs, 'w') as f:
    f.write(content)
print('[patch-whisper] build.rs patched successfully')
" "$BUILD_RS" "$PATCHES_DIR/build_rs_old.txt" "$PATCHES_DIR/build_rs_new.txt"
fi

# ── 2) Append Vulkan FFI to bindings.rs ──────────────────────────────────────
if grep -q "ggml_backend_vk_buffer_type" "$BINDINGS_RS" 2>/dev/null; then
    echo "[patch-whisper] Vulkan FFI already present"
else
    echo "[patch-whisper] Adding Vulkan FFI to bindings.rs ..."
    cat "$PATCHES_DIR/vulkan_ffi.txt" >> "$BINDINGS_RS"
    echo "[patch-whisper] Vulkan FFI added"
fi

echo "[patch-whisper] Done."
