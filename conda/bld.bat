@echo on
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

REM Confirm ninja is in path
where ninja || exit /b 1

REM Use Ninja generator explicitly
set "CMAKE_GENERATOR=Ninja"

REM Build the package using pip + scikit-build
%PYTHON% -m pip install . --no-deps --no-build-isolation -vv || exit /b 1