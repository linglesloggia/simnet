import sys
import os
import argparse
import pickle
import subprocess


class ConfigScenary():
    def __init__(self):
        self.num_slices = 1
        self.mimo = "SU"
        self.bs_name = "BS-1"
        self.name_sched = "round robin"
        self.rMCS = False
        self.longcp = False
        self.band = "n258"
        self.ul = False
        self.dl = True
        self.time_sim = 100

        self.name_slice = ["SL-1"]
        self.numerology = [0]

        self.sym_slot = [14]
        self.nresdl = [100]
        self.nresul = [0]
        self.name_resdl = ["PRB"]
        self.name_resul = ["PRB"]

        self.slice_uegr = [[0]]
        self.ugr_not_assigned = []

        self.channel_type = "random or fixed"
        self.file_channel = None
        self.channel_mode = "Random"
        self.val_1 = -10
        self.val_2 = 100
        self.loss_prob = 0

        self.trgen_type = ["poisson"]
        self.inter_arrival = [1]
        self.pkt_size = [300]
        self.burst_size = [1]
        self.size_dist = ["Exponential"]

        self.max_len = [0]
        self.keep_pkts = [False]
        self.last_k = [100]

        self.nuegroups = 1
        self.name_uegroup = ["UG-1"]
        self.uegr_par1 = [60]
        self.num_ues = [5]

config_file = "/content/drive/MyDrive/simnet/extensions/sim5gnr/data/config.pickle"

def load_config():
    if os.path.exists(config_file):
        with open(config_file, "rb") as fp:
            return pickle.load(fp)
    else:
        return ConfigScenary()

def save_config(conf):
    with open(config_file, "wb") as fp:
        pickle.dump(conf, fp)

def set_basestation(conf, mimo, num_slices, band, name, rMCS, longcp, ul, dl, time_sim):
    conf.mimo = mimo
    conf.bs_name = name
    conf.band = band
    conf.rMCS = eval(rMCS)
    conf.longcp = eval(longcp)
    conf.ul = eval(ul)
    conf.dl = eval(dl)
    conf.time_sim = eval(time_sim)
    configure_slices(conf, num_slices)

def configure_slices(conf, num_slices):
    conf.num_slices = int(num_slices)
    conf.name_slice = ["SL-"+str(i+1) for i in range(int(num_slices))]
    conf.numerology = [0] * int(num_slices)
    conf.sym_slot = [14] * int(num_slices)
    conf.nresdl = [100] * int(num_slices)
    conf.nresul = [0] * int(num_slices)
    conf.name_resdl = ["PRB"] * int(num_slices)
    conf.name_resul = ["PRB"] * int(num_slices)
    conf.slice_uegr = [[i] for i in range(int(num_slices))]

    conf.name_uegroup = ["UG-"+str(i+1) for i in range(int(num_slices))]
    conf.uegr_par1 = [50] * int(num_slices)
    conf.num_ues = [5] * int(num_slices)
    conf.trgen_type = ["poisson"] * int(num_slices)
    conf.inter_arrival = [1] * int(num_slices)
    conf.pkt_size = [300] * int(num_slices)
    conf.burst_size = [1] * int(num_slices)
    conf.size_dist = ["Exponential"] * int(num_slices)
    conf.max_len = [0] * int(num_slices)
    conf.keep_pkts = [False] * int(num_slices)
    conf.last_k = [100] * int(num_slices)
    conf.ugr_not_assigned = []

def set_uegroup(conf, index, name, par1, num_ues):
    index = int(index)
    if len(conf.name_uegroup) <= index:
        conf.name_uegroup.extend(["UG-"+str(i+1) for i in range(len(conf.name_uegroup), index+1)])
        conf.uegr_par1.extend([50] * (index + 1 - len(conf.uegr_par1)))
        conf.num_ues.extend([5] * (index + 1 - len(conf.num_ues)))

    conf.name_uegroup[index] = name
    conf.uegr_par1[index] = int(par1)
    conf.num_ues[index] = int(num_ues)

def set_resources(conf, slice_index, namedl, nameul, nresdl, nresul, sym_slot):
    slice_index = int(slice_index)
    if len(conf.name_slice) <= slice_index:
        configure_slices(conf, slice_index + 1)

    conf.sym_slot[slice_index] = int(sym_slot)
    conf.nresdl[slice_index] = int(nresdl)
    conf.nresul[slice_index] = int(nresul)
    conf.name_resdl[slice_index] = namedl
    conf.name_resul[slice_index] = nameul

def run_simulation(conf, debug):
    file_config = config_file
    with open(file_config, "wb") as fp:
        pickle.dump(conf, fp)

    proc = subprocess.Popen(['python3', '/content/drive/MyDrive/simnet/extensions/sim5gnr/gui/mk_simsetup.py', file_config, "True", debug], stdout=subprocess.PIPE)
    try:
        outs, errs = proc.communicate(timeout=600)
        print(outs.decode())
    except subprocess.TimeoutExpired:
        proc.kill()
        print("Simulation timeout")

def view_config(conf):
    file_config = config_file
    with open(file_config, "wb") as fp:
        pickle.dump(conf, fp)

    proc = subprocess.Popen(['python3', '/content/drive/MyDrive/simnet/extensions/sim5gnr/gui/mk_simsetup.py', file_config, "False", "True"], stdout=subprocess.PIPE)
    try:
        outs, errs = proc.communicate(timeout=300)
        print(outs.decode())
    except subprocess.TimeoutExpired:
        proc.kill()
        print("View configuration timeout")

def main(args):
    conf = load_config()

    if args.set_basestation:
        set_basestation(conf, *args.set_basestation)
    if args.set_uegroup:
        set_uegroup(conf, *args.set_uegroup)
    if args.set_resources:
        set_resources(conf, *args.set_resources)
    if args.run_simulation:
        run_simulation(conf, args.run_simulation)
    if args.view_config:
        view_config(conf)

    save_config(conf)


def run_command(command):

    #print('-- DEBUG --')
    parser = argparse.ArgumentParser(description='5G Python Wireless Simulator Command Line Interface')

    parser.add_argument('--set-basestation', nargs=9, metavar=('mimo', 'num_slices', 'band', 'name', 'rMCS', 'longcp', 'ul', 'dl', 'time_sim'), help='Set base station configuration')
    parser.add_argument('--set-uegroup', nargs=4, metavar=('index', 'name', 'par1', 'num_ues'), help='Set user group configuration')
    parser.add_argument('--set-resources', nargs=6, metavar=('slice_index', 'namedl', 'nameul', 'nresdl', 'nresul', 'sym_slot'), help='Set resources configuration')
    parser.add_argument('--run-simulation', metavar='debug', help='Run the simulation')
    parser.add_argument('--view-config', action='store_true', help='View the current configuration')

    args = parser.parse_args(command)
    main(args)

