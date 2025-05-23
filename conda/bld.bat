@echo on
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
python -m pip install . --no-deps --no-build-isolation -vv