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



REM Build the package using pip + scikit-build
%PYTHON% -m pip install . --no-deps --no-build-isolation -vv || exit /b 1