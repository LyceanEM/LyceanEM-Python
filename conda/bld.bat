@echo on
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

REM Confirm ninja is in path
where ninja || exit /b 1

REM Use Ninja generator explicitly
set "CMAKE_GENERATOR=Ninja"
REM Determine Python version
for /f %%v in ('%PYTHON% -c "import sys; print(f'python{sys.version_info.major}{sys.version_info.minor}')"') do set PYTHON_LIB=%%v.lib

REM Get Python version (e.g., python39)
for /f %%v in ('%PYTHON% -c "import sys; print(f'python{sys.version_info.major}{sys.version_info.minor}')"') do set PYTHON_LIB_NAME=%%v

REM Set full path to the expected .lib file
set "PYTHON_LIB=%PREFIX%\Library\libs\python%PY_VER:~0,1%%PY_VER:~2,1%.lib"

REM Check if it exists
if not exist "%PYTHON_LIB%" (
    echo ERROR: Required file "%PYTHON_LIB%" not found!
    echo This file is needed to link your C++/CUDA extension with Python.
    exit /b 1
)
else (
    echo Found Python library: %PYTHON_LIB%
)
set "CMAKE_ARGS=-DPYTHON_LIBRARY=%PYTHON_LIB%"


REM Build the package using pip + scikit-build
%PYTHON% -m pip install . --no-deps --no-build-isolation -vv || exit /b 1