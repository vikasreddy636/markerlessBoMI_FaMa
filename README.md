TESTED with python>=3.7

# UBUNTU and MAC installation steps:

step 0 --> open a terminal and cd to the home folder:

``` 
    $ cd
``` 

step 1 --> [if not done yet] install pip and venv:

``` 
    $ sudo apt-get update
    $ sudo apt install python3-pip
    $ sudo apt install build-essential libssl-dev libffi-dev python3-dev
    $ sudo apt install python3-venv
``` 

step 2 --> create a virtual environment called BoMI:

``` 
    $ python3 -m venv BoMI
``` 

step 3 --> activate virtual enviroment:

``` 
    $ source BoMI/bin/activate
``` 

step 4 --> upgrade pip and install all the packages nedded for markerlessBoMI:

``` 
    $ pip install --upgrade pip
    $ pip install git+https://github.com/MoroMatteo/markerlessBoMI_FaMa.git
``` 

step 5 --> istall tkinter:

``` 
    $ sudo apt install python3-tk
``` 

step 6 --> clone the github repository:

``` 
    $ git clone https://github.com/MoroMatteo/markerlessBoMI_FaMa.git
``` 

step 7 --> cd in the correct folder and run main_reaching.py:

``` 
    $ cd markerlessBoMI_FaMa/
    $ python3 main_reaching.py
``` 

step 8 --> follow the steps in the GUI (see below after WINDOWS installation steps)

# WINDOWS installation steps:

