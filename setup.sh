virtualenv venv
source setup_git_filters.sh
printf '\nexport PYTHONPATH=$PYTHONPATH:%s\n' $PWD >> venv/bin/activate
source venv/bin/activate
pip install -r requirements.txt
echo "Plato Setup Complete! (As long as there're no errors above)"
