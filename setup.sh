virtualenv venv
printf '\nexport PYTHONPATH=$PYTHONPATH:%s\n' $PWD >> venv/bin/activate
source venv/bin/activate
pip install -r requirements.txt
echo "Setup Complete! (As long as there're no errors above)"
