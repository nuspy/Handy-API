@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
set "PATH=%USERPROFILE%\.cargo\bin;%USERPROFILE%\scoop\shims;%PATH%"
set "_CL_="
set "CL="
set "LIBCLANG_PATH=C:\Users\atain\AppData\Local\Programs\Python\Python310\lib\site-packages\clang\native"
set "VULKAN_SDK=C:\Users\atain\scoop\apps\vulkan\current"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe"
set "CMAKE_CUDA_ARCHITECTURES=120"
set "GGML_CUDA_ARCHITECTURES=120"
set "NVCC_PREPEND_FLAGS=-allow-unsupported-compiler"
set "CMAKE_GENERATOR=Ninja"
set "CMAKE_POLICY_VERSION_MINIMUM=3.5"
set "CMAKE_MAKE_PROGRAM=C:\Users\atain\scoop\apps\ninja\current\ninja.exe"
set "RUSTUP_TOOLCHAIN=stable"
set "RUST_MIN_STACK=16777216"
cd /d "C:\Projects\Handy-API\src-tauri"
cargo check --release 2>&1
