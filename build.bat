@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
set "LIBCLANG_PATH=C:\Program Files\LLVM\bin"
set "VULKAN_SDK=C:\VulkanSDK\1.4.341.1"
set "CMAKE_GENERATOR=Ninja"
set "CMAKE_POLICY_VERSION_MINIMUM=3.5"
set "CMAKE_MAKE_PROGRAM=C:\Users\hpome\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe\ninja.exe"
cd /d "D:\AI\Handy-API"
bun x @tauri-apps/cli build 2>&1
