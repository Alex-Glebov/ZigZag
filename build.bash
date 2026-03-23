source ../.venv/bin/activate
echo build package "$PWD" 
poetry build
echo
echo
echo --- example of installation ---
echo --- editable  version  --
echo pip install  -e "$PWD"
echo --- just as package --
echo pip install    "$PWD"
echo pip install    "$PWD"
echo --force-reinstal use it for force reinstall  package  
deactivate
read yn
