#! /usr/bin/bash
echo "If you Want to train model from scratch ENTER 'training' as input else for only check the app ENTER 'no' :"
read userinput
if [[ $userinput == "training" ]]
then
    python main.py
 elif [[ $userinput == "no" ]]
then
    python app.py
else
    echo "Please Read the instruction carefully! "
fi