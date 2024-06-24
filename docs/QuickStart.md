![PyWiSim logo](diagrams/PyWiSim-logo260px.png)

[Back to README](../README.md)

# Quick start

This guide provides some brief instructions to install and test the simulator, as a first step to extend its capabilities towards different protocol, standard or algorithm implementations.


## Installation
1. Clone the project.
```
    cd <your_install_directory>
    git clone https://gitlab.fing.edu.uy/vagonbar/simnet  
```

2. Adjust the PYTHONPATH variable.
In Linux:
```
    export PYTHONPATH=$PYTHONPATH:<your_install_directory>/simnet
```

To make this change permanent, add the former command to your ~/.bashrc file and issue the command
```
    source .bashrc
```

To adjust the `PYTHONPATH` variable under other operating systems, please see [Installation](docs/Installation.md)


## Run the tests

After setting the `PYTHONPATH` variable, you may run tests for a simple example in this way:
```
    cd <your directory/PyWiSim/extensions/simplesim
    python3 qa_simulator.py
```


## Directory structure

The project directory structure is designed to allow for other simulation models or software extension in a orderly fashion, mainly by creating subdirectories in the `extensions` or `models` directories. In this way, PyWiSim may be extended without touching the `libsimnet` library, just by overwriting its classes or adding new classes, which will reside in a subdirectory of the `extensions` of `models` directories. The PyWiCh channel simulator is an example of a channel models which may be run within the PyWiSim simulator.The PyWiCh channel simulator may be found in the `models/channel/pywich` directory.

- `docs`: documentation.
- `html` : [pydoctor](https://pydoctor.readthedocs.io/en/latest/) code documentation.
    - `html/index.html` : main page for code documentation.
- `libsimnet` : the main library.
- `extensions` : code for different simulator implementations.
    - `simplesim` : extension for a simple simulator implementation.
- `models` : code for different entities simulation models.
    - `channel` : different channel models.
        - `filechannel` : channel states taken from a CSV file.
        - `pywich` : extension for the PyWiCh channel simulator.
    - `scheduler` : different scheduler models.
        - `rrscheduler` : a round robin scheduler.
        - `simplesched` : a simple random based scheduler.
- `sandbox` : a testbed for different functionalities.


[Back to README](../README.md)

