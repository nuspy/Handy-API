@echo off
rem -vcvars_ver=14.44: forza il toolset MSVC 14.44 (VS 2022) invece del 14.51
rem (VS 18), perche' nvcc 13.1 non supporta 14.51 (cudafe++ crasha). Il 14.44 e'
rem installato a fianco ed e' compatibile con CUDA 13.1.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44 >nul 2>&1
set "PATH=%USERPROFILE%\.cargo\bin;%USERPROFILE%\scoop\shims;%PATH%"
rem MSVC cl.exe accoda _CL_/CL a OGNI invocazione: se _CL_ contiene
rem /std:c++20 (impostata a livello User sul sistema) rompe la compilazione
rem dei file .c di ggml con "D8016: /std:c11 e /std:c++20 incompatibili".
rem Le azzeriamo solo per questo build (lo scope persistente resta intatto).
set "_CL_="
set "CL="
rem LLVM 22 di scoop trascina header glibc/mingw che bindgen pickerebbe al
rem posto di MSVC -> bindings con _G_fpos_t/_IO_FILE che non matchano.
rem Usiamo la libclang 18.1.1 fornita dal pacchetto pip 'libclang' (solo DLL,
rem niente header) cosi' clang ricade sugli header del Windows SDK / MSVC.
set "LIBCLANG_PATH=C:\Users\atain\AppData\Local\Programs\Python\Python310\lib\site-packages\clang\native"
set "VULKAN_SDK=C:\Users\atain\scoop\apps\vulkan\current"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe"
rem Architettura CUDA per la build di whisper.cpp (whisper-rs/cuda):
rem 120 = Blackwell (RTX 5090, sm_120). Senza questo CMake potrebbe targetizzare
rem un'arch di default sbagliata o fallire l'autodetect.
set "CMAKE_CUDA_ARCHITECTURES=120"
set "GGML_CUDA_ARCHITECTURES=120"
rem VS 18 (MSVC 14.51) e' piu' nuovo di quanto nvcc 13.1 supporti ufficialmente
rem (solo VS 2019-2022). Bypassiamo il controllo di versione in host_config.h.
set "NVCC_PREPEND_FLAGS=-allow-unsupported-compiler"
set "CMAKE_GENERATOR=Ninja"
set "CMAKE_POLICY_VERSION_MINIMUM=3.5"
set "CMAKE_MAKE_PROGRAM=C:\Users\atain\scoop\apps\ninja\current\ninja.exe"
set "CARGO_TARGET_DIR=C:\B"
rem WHISPER_DONT_GENERATE_BINDINGS=1 era una scorciatoia per 0.13.1 con
rem patch MSVC custom (vedi patch-whisper.sh, ora rimosso). Con 0.15.0
rem bindgen deve rigenerare i binding su MSVC.
set /p TAURI_SIGNING_PRIVATE_KEY=<"C:\Projects\Handy-API\.tauri-keys"
set /p TAURI_SIGNING_PRIVATE_KEY_PASSWORD=<"C:\Projects\Handy-API\.tauri-keys-password"
set "RUSTUP_TOOLCHAIN=stable"
set "RUST_MIN_STACK=16777216"
cd /d "C:\Projects\Handy-API"

rem patch-whisper.sh era specifico per whisper-rs-sys 0.13.1 (workaround
rem libclang 22). Dal merge upstream usiamo whisper-rs-sys 0.15.0 con bindings
rem rigenerati a build-time — il patch non e' piu' necessario.

bun x @tauri-apps/cli build --bundles nsis 2>&1
