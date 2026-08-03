@echo off
setlocal

where nvcc >nul 2>nul
if errorlevel 1 (
    echo nvcc was not found. Install the CUDA Toolkit and add its bin directory to PATH.
    exit /b 1
)

if not defined VSCMD_VER (
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) else (
        echo Visual Studio 2022 C++ build tools were not found.
        echo Run this script from an x64 Native Tools Command Prompt or install VS 2022 Build Tools.
        exit /b 1
    )
)

if not defined CUDA_ARCH set "CUDA_ARCH=sm_120"
set "NVCC_COMMON=-shared -Xcompiler /MD --use_fast_math -arch=%CUDA_ARCH%"

echo Compiling fp8_kernels.cu for %CUDA_ARCH%...
nvcc %NVCC_COMMON% -o qwen3_infer\csrc\fp8_kernels.dll qwen3_infer\csrc\fp8_kernels.cu
if errorlevel 1 (
    echo fp8_kernels.cu compilation failed.
    exit /b 1
)

echo Compiling fused_kernels.cu for %CUDA_ARCH%...
nvcc %NVCC_COMMON% -o qwen3_infer\csrc\fused_kernels.dll qwen3_infer\csrc\fused_kernels.cu
if errorlevel 1 (
    echo fused_kernels.cu compilation failed.
    exit /b 1
)

echo Done.
endlocal