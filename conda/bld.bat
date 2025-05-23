@echo on
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

REM Confirm ninja is in path
where ninja || exit /b 1

REM Use Ninja generator explicitly
set "CMAKE_GENERATOR=Ninja"
REM Determine Python version
for /f %%v in ('%PYTHON% -c "import sys; print(f'python{sys.version_info.major}{sys.version_info.minor}')"') do set PYTHON_LIB=%%v.lib

REM Check if pythonXXX.lib exists
set "PYTHON_LIB=%PREFIX%\libs\python39.lib"
if not exist "%PYTHON_LIB%" (
    echo ERROR: Required file "%PYTHON_LIB%" not found!
    echo This file is needed to link your C++/CUDA extension with Python.
    exit /b 1
)
else (
    echo Found Python library: %PYTHON_LIB%
)
if not exist "%LIBRARY_LIB%\%PYTHON_LIB%" (
  echo ERROR: Required file "%LIBRARY_LIB%\%PYTHON_LIB%" not found!
  echo This file is needed to link your C++/CUDA extension with Python.
  exit /b 1
)
REM Build the package using pip + scikit-build
%PYTHON% -m pip install . --no-deps --no-build-isolation -vv || exit /b 1