@echo off
setlocal

set "CMAKE_GENERATOR=Visual Studio 17 2022"
set "CMAKE_GENERATOR_TOOLSET=v143"
set "VSINSTALLDIR=C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
set "INCLUDE=%BUILD_PREFIX%\Library\include;%BUILD_PREFIX%\Library\include"
set "LIB=%BUILD_PREFIX%\Library\lib;%BUILD_PREFIX%\Library\lib"
set "CMAKE_PREFIX_PATH=%BUILD_PREFIX%\Library"

call "%VSINSTALLDIR%\VC\Auxiliary\Build\vcvarsall.bat" x64

:: Your build commands here
python  -m pip install . -vv
if errorlevel 1 exit 1
endlocal


