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

step 0 --> download Python3 at this link https://www.python.org/downloads/ (python version >= 3.7)

step 1 --> open a command window (terminal) as root ("amministratore") and type 

``` 
    $ cd
``` 

step 2 --> install pip and virtualenv

``` 
    $ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $ python get-pip.py
    $ pip install virtualenv
``` 

step 3 --> create virtualenv (named BoMI) and activate it

``` 
    $ python3 -m venv BoMI
    $ BoMI\Scripts\activate
``` 

step 4 --> enable long path (https://www.howtogeek.com/266621/how-to-make-windows-10-accept-file-paths-over-260-characters/) --> remember to launch regedit as root ("amministratore")

step 5 --> Upgrade pip and download all the following packages (in the terminal):

``` 
    $ pip install --upgrade pip
    $ pip install numpy
    $ pip install pandas
    $ pip install matplotlib
    $ pip install pygame
    $ pip install pyautogui
    $ pip install tensorflow
    $ pip install mediapipe
    $ pip install scipy
    $ pip install sklearn
``` 

step 6 --> download Visual Studio Code (https://code.visualstudio.com/download)

Step 7 --> Download the repository from the current github page and open it as a project in Visual Studio Code

step 8 --> Set the correct python interpreter in Visual Studio Code (the one of the virtual environment created - BoMI\Bin\python)

step 9 --> eventually [not always] there is the possibility that it is necessary to do the steps described here https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads

step 10 --> On Visual Studio Code run the file main_reaching.py

# Graphical User Interface
