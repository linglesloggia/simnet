![PyWiSim logo](diagrams/PyWiSim-logo260px.png)

[Back to README](../README.md)



# Installation

### Clone the project
To clone the project to your local machine:
```
    cd <your_install_directory>
    git clone https://gitlab.fing.edu.uy/vagonbar/simnet
```
After cloning the project, your must adjust the search path so that PyWiSim can access its libraries and resources, and also the libraries and resources of complementary software if needed, e.g. PyWiCh for a channel model. Once these adjustments are made, you can run the tests.

### Adjust search path

The preferred method to adjust the search path to ensure access to libraries and resources is to set the paths in the PYTHONPATH environment variable. This ensures access without altering the code, and allows the user to install PyWiSim, PyWiCh or any other related software in the user's preferred locaions.


#### In Linux

To set the `PYTHONPATH` variable for a single time (you will be required to do this each time you want to run the tests):
```
    export PYTHONPATH=$PYTHONPATH:<your_install_directory/simnet
```
You can add other search paths, for example to access PyWiCh, adding the PyWiCh installation directory to the PYTHONPATH variable in the same way. To make this change permanent, you can add this command to your `.bashrc` file in your home directory. After modifying the `~/.bashrc` file, please issue the command:
```
    source ~/.bashrc
```
so that it takes effect. After this, you may run tests.


#### In MS Windows

Please follow the instructions on these references:

- TutorialsPoint. [How to set python environment variable PYTHONPATH on Windows?](https://www.tutorialspoint.com/How-to-set-python-environment-variable-PYTHONPATH-on-Windows)
- Python documentation. Using Python on Windows. [4.6.1. Excursus: Setting environment variables.](https://docs.python.org/3/using/windows.html#excursus-setting-environment-variables) 

#### In macOS

Please see this reference:

- Python documentation. Using Python on a Mac. [5.1.3. Configuration.](https://docs.python.org/3/using/mac.html#configuration)


### Run the tests

After setting the `PYTHONPATH` variable, you may run tests in this way:
```
    cd <your directory/simnet/extensions/simplesim
    python3 qa_simulator.py
```

QA stands for "Quality Assurance"; test files are named with prefix "qa_" following the [GNU Radio](https://www.gnuradio.org/) tradition.

In the `extensions` directory you may find different implementations of simulators as extensions to the PyWiSim library, i.e. `simplesim`, a simple simulator implementaion; `qa_` modules provide tests for this simple implementation.

In the `models` directory you may find different implementations of simulation for a specific entity, such as a telecommunications channel model, or a scheduler to assign telecommunications resources to user equipments. For example, in `models/channel/pywich` you may find simulations based on different PyWiCh channel simulators, with their own `qa_` modules for testing.


[Back to README](../README.md)


