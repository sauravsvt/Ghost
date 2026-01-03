# Ghost C++ Backend Build Configuration
#
# This CMakeLists.txt configures the build for libghost, the native
# Bi-Mamba 1.58-bit inference backend.
#
# Build Instructions:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build . --config Release
#
# Output:
#   Linux:   libghost.so
#   Windows: ghost.dll
#   macOS:   libghost.dylib

cmake_minimum_required(VERSION 3.16)
project(GhostEngine VERSION 1.0.0 LANGUAGES CXX)

# C++17 required for std::optional, structured bindings, etc
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Release optimizations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Aggressive optimization flags for CPU inference
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG /arch:AVX2")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /DDEBUG")
endif()

# ============================================================================
# Core Ghost Library
# ============================================================================
add_library(ghost SHARED
    ghost_lib.cpp
)

# Platform-specific settings
if(WIN32)
    set_target_properties(ghost PROPERTIES
        OUTPUT_NAME "ghost"
        SUFFIX ".dll"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
elseif(APPLE)
    set_target_properties(ghost PROPERTIES
        OUTPUT_NAME "ghost"
        SUFFIX ".dylib"
    )
else()
    set_target_properties(ghost PROPERTIES
        OUTPUT_NAME "ghost"
        SUFFIX ".so"
    )
endif()

# ============================================================================
# Optional: Link against BitNet.cpp
# ============================================================================
option(USE_BITNET_CPP "Link against BitNet.cpp for native 1.58-bit inference" OFF)

if(USE_BITNET_CPP)
    # Assumes BitNet.cpp is installed to /usr/local or CMAKE_PREFIX_PATH
    find_package(BitNet QUIET)
    if(BitNet_FOUND)
        target_link_libraries(ghost PRIVATE BitNet::bitnet)
        target_compile_definitions(ghost PRIVATE HAVE_BITNET_CPP)
        message(STATUS "BitNet.cpp found and linked")
    else()
        message(WARNING "BitNet.cpp not found. Using fallback implementation.")
    endif()
endif()

# ============================================================================
# Optional: Link against llama.cpp
# ============================================================================
option(USE_LLAMA_CPP "Link against llama.cpp for transformer inference" OFF)

if(USE_LLAMA_CPP)
    find_library(LLAMA_LIB llama HINTS /usr/local/lib ${CMAKE_PREFIX_PATH}/lib)
    find_path(LLAMA_INCLUDE llama.h HINTS /usr/local/include ${CMAKE_PREFIX_PATH}/include)
    
    if(LLAMA_LIB AND LLAMA_INCLUDE)
        target_include_directories(ghost PRIVATE ${LLAMA_INCLUDE})
        target_link_libraries(ghost PRIVATE ${LLAMA_LIB})
        target_compile_definitions(ghost PRIVATE HAVE_LLAMA_CPP)
        message(STATUS "llama.cpp found: ${LLAMA_LIB}")
    else()
        message(WARNING "llama.cpp not found. Using fallback implementation.")
    endif()
endif()

# ============================================================================
# Optional: CUDA Support for Mamba kernels
# ============================================================================
option(USE_CUDA "Enable CUDA for Mamba selective scan kernels" OFF)

if(USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        target_compile_definitions(ghost PRIVATE HAVE_CUDA)
        target_link_libraries(ghost PRIVATE CUDA::cudart)
        message(STATUS "CUDA enabled for Mamba kernels")
    else()
        message(WARNING "CUDA not found. Using CPU-only implementation.")
    endif()
endif()

# ============================================================================
# Threading support
# ============================================================================
find_package(Threads REQUIRED)
target_link_libraries(ghost PRIVATE Threads::Threads)

# ============================================================================
# Math library (required for exp, etc)
# ============================================================================
if(NOT WIN32)
    target_link_libraries(ghost PRIVATE m)
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS ghost
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "Ghost Engine Build Configuration:")
message(STATUS "  Build Type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   C++${CMAKE_CXX_STANDARD}")
message(STATUS "  BitNet.cpp:     ${USE_BITNET_CPP}")
message(STATUS "  llama.cpp:      ${USE_LLAMA_CPP}")
message(STATUS "  CUDA:           ${USE_CUDA}")
message(STATUS "========================================")
message(STATUS "")
