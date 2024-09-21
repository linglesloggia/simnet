import sys
from AppCmd import *
sys.path.append('/home/lingles/ownCloud/Summer_school/simnet')

pywisim(['--set-basestation', 'SU', '1', 'n258', 'BS-1', 'False', 'False', 'False', 'True', '500', 'ground_base_learn', "DQN_channel030001", 'Random'])
pywisim(['--set-uegroup', '0', 'UG-1', '60', '200', '2', 'DQN_trfgen030001', '3'])
pywisim(['--set-uegroup', '1', 'UG-2', '60', '300', '1', 'DQN_trfgen020001', '3'])
pywisim(['--set-uegroup', '2', 'UG-2', '60', '100', '5', 'DQN_trfgen010001', '4'])
pywisim(['--set-resources', '0', 'PRB', 'PRB', '5', '0', '14'])

pywisim(['--run-simulation', 'True'])