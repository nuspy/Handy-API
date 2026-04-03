cmake_minimum_required(VERSION 3.22.1)
project("handyvoice" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================
# whisper.cpp integration
# ==============================================================
# Clone whisper.cpp into this directory:
#   git submodule add https://github.com/ggerganov/whisper.cpp.git whisper.cpp
#
# The submodule provides the full whisper + ggml source tree.

set(WHISPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp)

if(EXISTS ${WHISPER_DIR}/CMakeLists.txt)
    # Build whisper as a static library
    set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    add_subdirectory(${WHISPER_DIR} whisper_build)
    set(WHISPER_AVAILABLE TRUE)
    message(STATUS "whisper.cpp found and will be built")
else()
    set(WHISPER_AVAILABLE FALSE)
    message(WARNING "whisper.cpp not found at ${WHISPER_DIR}. STT will be unavailable.")
    message(WARNING "Run: git submodule add https://github.com/ggerganov/whisper.cpp.git android-app/app/src/main/cpp/whisper.cpp")
endif()

# ==============================================================
# llama.cpp integration (for Qwen3-TTS inference)
# ==============================================================
# Clone llama.cpp into this directory:
#   git submodule add https://github.com/ggerganov/llama.cpp.git llama.cpp

set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

if(EXISTS ${LLAMA_DIR}/CMakeLists.txt)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    add_subdirectory(${LLAMA_DIR} llama_build)
    set(LLAMA_AVAILABLE TRUE)
    message(STATUS "llama.cpp found and will be built")
else()
    set(LLAMA_AVAILABLE FALSE)
    message(WARNING "llama.cpp not found at ${LLAMA_DIR}. TTS will be unavailable.")
    message(WARNING "Run: git submodule add https://github.com/ggerganov/llama.cpp.git android-app/app/src/main/cpp/llama.cpp")
endif()

# ==============================================================
# JNI Bridge Library
# ==============================================================
add_library(handyvoice SHARED
    whisper-jni.cpp
    llama-jni.cpp
)

target_include_directories(handyvoice PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link Android log
find_library(log-lib log)
target_link_libraries(handyvoice ${log-lib})

# Link whisper if available
if(WHISPER_AVAILABLE)
    target_link_libraries(handyvoice whisper)
    target_compile_definitions(handyvoice PRIVATE WHISPER_AVAILABLE=1)
    target_include_directories(handyvoice PRIVATE ${WHISPER_DIR}/include)
endif()

# Link llama if available
if(LLAMA_AVAILABLE)
    target_link_libraries(handyvoice llama ggml)
    target_compile_definitions(handyvoice PRIVATE LLAMA_AVAILABLE=1)
    target_include_directories(handyvoice PRIVATE ${LLAMA_DIR}/include)
endif()
