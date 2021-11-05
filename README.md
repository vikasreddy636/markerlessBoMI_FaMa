TESTED with python>=3.7

# UBUNTU and MAC installation steps:

step 0 --> open a terminal and cd to the home folder:

``` 
    $ cd
``` 

step 1 --> [if not done yet] install pip and venv

``` 
    $ sudo apt-get update
    $ sudo apt install python3-pip
    $ sudo apt install build-essential libssl-dev libffi-dev python3-dev
    $ sudo apt install python3-venv
``` 

step 1 --> create a virtual environment ($ python3 -m venv $path/to/place/env$)

step 2 --> activate virtual enviroment -- $ source $path/to/place/env$/bin/activate

step 3 --> upgrade pip -- $ pip install --upgrade pip

step 4 --> pip install git+https://github.com/MoroMatteo/markerlessBoMI_FaMa.git

step 5 --> open terminal and run main_reaching.py
