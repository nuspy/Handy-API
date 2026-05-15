@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
set "PATH=%USERPROFILE%\.cargo\bin;%USERPROFILE%\scoop\shims;%PATH%"
set "LIBCLANG_PATH=C:\Users\atain\scoop\apps\llvm\22.1.0\bin"
set "VULKAN_SDK=C:\Users\atain\scoop\apps\vulkan\current"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe"
set "CMAKE_GENERATOR=Ninja"
set "CMAKE_POLICY_VERSION_MINIMUM=3.5"
set "CMAKE_MAKE_PROGRAM=C:\Users\atain\scoop\apps\ninja\current\ninja.exe"
set "CARGO_TARGET_DIR=C:\B"
set "WHISPER_DONT_GENERATE_BINDINGS=1"
set /p TAURI_SIGNING_PRIVATE_KEY=<"C:\Projects\Handy-API\.tauri-keys"
set "TAURI_SIGNING_PRIVATE_KEY_PASSWORD="
set "RUSTUP_TOOLCHAIN=stable"
set "RUST_MIN_STACK=16777216"
cd /d "C:\Projects\Handy-API"

echo Applying whisper-rs-sys patches...
bash patch-whisper.sh
if errorlevel 1 (
    echo [WARNING] patch-whisper.sh failed, build may fail
)

bun x @tauri-apps/cli build --bundles nsis 2>&1
