import sys
import os
import argparse
import pickle
import subprocess

import matplotlib.pyplot as plt
import numpy as np

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

        self.slice_uegr = [[0,1]]
        self.ugr_not_assigned = []

        self.channel_type = "random or fixed"
        self.file_channel = None
        self.channel_mode = "Random"
        self.val_1 = -10
        self.val_2 = 100
        self.loss_prob = 0

        self.trgen_type = ["poisson"]*2
        self.inter_arrival = [1]*2
        self.pkt_size = [300]*2
        self.burst_size = [1]*2
        self.size_dist = ["Exponential"]*2

        self.max_len = [0]
        self.keep_pkts = [False]
        self.last_k = [100]

        self.nuegroups = 2
        self.name_uegroup = ["UG-1","UG-2"]#,["UG-2"]]
        self.uegr_par1 = [60,60] #* 2#,[60]]
        self.num_ues = [5,5]#,[5]]

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

def set_slice(conf, slice_index, slice_name, numerology, user_groups, num_uegroups, sym_slot, nresdl, nresul, namedl, nameul):
    slice_index = int(slice_index)
    numerology = int(numerology)
    num_uegroups = int(num_uegroups)
    sym_slot = int(sym_slot)
    nresdl = int(nresdl)
    nresul = int(nresul)
    user_groups = user_groups.split(',')

    if len(conf.name_slice) <= slice_index:
        configure_slices(conf, slice_index + 1)

    conf.name_slice[slice_index] = slice_name
    conf.numerology[slice_index] = numerology
    conf.sym_slot[slice_index] = sym_slot
    conf.nresdl[slice_index] = nresdl
    conf.nresul[slice_index] = nresul
    conf.name_resdl[slice_index] = namedl
    conf.name_resul[slice_index] = nameul

    conf.slice_uegr[slice_index] = []
    for user_group in user_groups:
        conf.slice_uegr[slice_index].append(conf.name_uegroup.index(user_group))

def set_basestation(conf, mimo, num_slices, band, name, rMCS, longcp, ul, dl, time_sim, name_sched, channel_type, channel_mode):
    conf.mimo = mimo
    conf.bs_name = name
    conf.band = band
    conf.rMCS = eval(rMCS)
    conf.longcp = eval(longcp)
    conf.ul = eval(ul)
    conf.dl = eval(dl)
    conf.time_sim = eval(time_sim)

    conf.name_sched = name_sched
    conf.channel_mode = channel_mode
    conf.channel_type = channel_type

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

    num_slices = 2
    conf.slice_uegr = [[0,1]]
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


def set_uegroup(conf, index, name, par1, pkt_size, inter_arrival, trgen_type, num_ues):
    index = int(index)
    #print(index, name, par1, num_ues)
    
    conf.name_uegroup[index] = name
    conf.uegr_par1[index] = int(par1)
    conf.num_ues[index] = int(num_ues)
    conf.pkt_size[index] = int(pkt_size)
    conf.inter_arrival[index] = int(inter_arrival)
    conf.trgen_type[index] = trgen_type


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

def process_data(file):
    data = []
    with open(file, 'r') as fp:
        for line in fp:
            data.append(eval(line.strip()))
    return data

def graph_data(times, values, title, ylabel):
    x = np.arange(len(times))
    fig, ax = plt.subplots()
    ax.bar(x, values, align='center', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Redondear los tiempos a enteros y mostrarlos cada 10 etiquetas
    rounded_times = [int(round(time)) for time in times]
    plt.xticks(x[::10], rounded_times[::10])  # Mostrar cada 10 etiquetas

    plt.tight_layout()
    plt.show()

def graph_subplots(data_list, times_list, title, ylabel, users):
    fig, axes = plt.subplots(len(data_list), 1, figsize=(10, 5 * len(data_list)), sharex=True)
    if len(data_list) == 1:
        axes = [axes]
    for ax, data, times, user in zip(axes, data_list, times_list, users):
        x = np.arange(len(times))
        ax.bar(x, data, align='center', alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} for User {user}")
        rounded_times = [int(round(time)) for time in times]
        ax.set_xticks(x[::10])
        ax.set_xticklabels(rounded_times[::10])
    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()

#Ajusto el tamaño de los datos, por si son distintos
def adjust_lengths(*lists):
    min_length = min(len(lst) for lst in lists)
    return [lst[:min_length] for lst in lists]

#Defino una nueva funcion para graficar los grupos
def graph_dataG(times, values1, values2, title, ylabel):
    times, values1, values2 = adjust_lengths(times, values1, values2)
    x = np.arange(len(times))
    fig, ax = plt.subplots()
    ax.plot(x, values1, marker='o', linestyle='-', alpha=0.7, label='UE 0 + UE 1')
    ax.plot(x, values2, marker='o', linestyle='-', alpha=0.7, label='UE 2 + UE 3')
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    rounded_times = [int(round(time)) for time in times]
    plt.xticks(x[::10], rounded_times[::10])

    plt.tight_layout()
    plt.show()


#Defino una funcion para sumar datos de dos usuarios
def sum_ues(data1, data2):
    return [x + y for x, y in zip(data1, data2)]


def process_and_graph_data(select, subselection, title):
    file_rec = '/content/drive/MyDrive/simnet/extensions/sim5gnr/data/run_results_rec.txt'
    file_tr = '/content/drive/MyDrive/simnet/extensions/sim5gnr/data/run_results_tr.txt'
    file_res = '/content/drive/MyDrive/simnet/extensions/sim5gnr/data/run_results_res.txt'
    file_queue = '/content/drive/MyDrive/simnet/extensions/sim5gnr/data/run_results_queue.txt'

    times_rec, bits_rec, bits_drop = [[]], [[]], [[]]
    times_tr, bits_sent, bits_lost = [[]], [[]], [[]]
    times_res, tbits, resources = [[]], [[]], [[]]
    times_delay, bits, pkts, delay = [[]], [[]], [[]], [[]]

    with open(file_rec, 'r') as fp:
        ue = 0
        for line in fp:
            data = eval(line)
            times_rec[ue] = [d[0] for d in data]
            bits_rec[ue] = [d[1] for d in data]
            bits_drop[ue] = [d[2] for d in data]
            ue += 1
            times_rec.append([])
            bits_rec.append([])
            bits_drop.append([])

    with open(file_tr, 'r') as fp:
        ue = 0
        for line in fp:
            data = eval(line)
            times_tr[ue] = [d[0] for d in data]
            bits_sent[ue] = [d[1] for d in data]
            bits_lost[ue] = [d[2] for d in data]
            ue += 1
            times_tr.append([])
            bits_sent.append([])
            bits_lost.append([])

    with open(file_res, 'r') as fp:
        ue = 0
        for line in fp:
            data = eval(line)
            times_res[ue] = [d[0] for d in data]
            tbits[ue] = [d[2] for d in data]
            resources[ue] = [d[1] for d in data]
            ue += 1
            times_res.append([])
            tbits.append([])
            resources.append([])

    with open(file_queue, 'r') as fp:
        ue = 0
        for line in fp:
            data = eval(line)
            times_delay[ue] = [d[0] for d in data]
            bits[ue] = [d[1] for d in data]
            pkts[ue] = [d[2] for d in data]
            delay[ue] = [d[3] for d in data]
            ue += 1
            times_delay.append([])
            bits.append([])
            pkts.append([])
            delay.append([])

    # Ensure the last empty lists are removed
    times_rec.pop()
    bits_rec.pop()
    bits_drop.pop()
    times_tr.pop()
    bits_sent.pop()
    bits_lost.pop()
    times_res.pop()
    tbits.pop()
    resources.pop()
    times_delay.pop()
    bits.pop()
    pkts.pop()
    delay.pop()


    #sumo los datos de los usuarios 0 y 1
    # Sumar los datos de ue=0 con ue=1 y ue=2 con ue=3
    times_sum = times_rec[0]  # Asumimos que los tiempos son los mismos
    bits_sum_01 = sum_ues(bits_rec[0], bits_rec[1])
    bits_sum_23 = sum_ues(bits_rec[2], bits_rec[3])
    bits_drop_sum_01 = sum_ues(bits_drop[0], bits_drop[1])
    bits_drop_sum_23 = sum_ues(bits_drop[2], bits_drop[3])
    
    bits_sent_sum_01 = sum_ues(bits_sent[0], bits_sent[1])
    bits_sent_sum_23 = sum_ues(bits_sent[2], bits_sent[3])
    bits_lost_sum_01 = sum_ues(bits_lost[0], bits_lost[1])
    bits_lost_sum_23 = sum_ues(bits_lost[2], bits_lost[3])
    
    tbits_sum_01 = sum_ues(tbits[0], tbits[1])
    tbits_sum_23 = sum_ues(tbits[2], tbits[3])
    resources_sum_01 = sum_ues(resources[0], resources[1])
    resources_sum_23 = sum_ues(resources[2], resources[3])
    
    bits_sum_01_queue = sum_ues(bits[0], bits[1])
    bits_sum_23_queue = sum_ues(bits[2], bits[3])
    pkts_sum_01 = sum_ues(pkts[0], pkts[1])
    pkts_sum_23 = sum_ues(pkts[2], pkts[3])
    delay_sum_01 = sum_ues(delay[0], delay[1])
    delay_sum_23 = sum_ues(delay[2], delay[3]) 



    # Graph based on the value of select
    if select == '0':
        graph_subplots(bits_rec, times_rec, title + " - Bits Received", "Bits", list(range(len(bits_rec))))
        if subselection == '1':
            graph_subplots(bits_drop, times_rec, title + " - Bits Dropped", "Bits", list(range(len(bits_rec))))
    elif select == '1':
        graph_subplots(bits_sent, times_tr, title + " - Bits Sent", "Bits", list(range(len(bits_sent))))
        if subselection == '1':
            graph_subplots(bits_lost, times_tr, title + " - Bits Lost", "Bits", list(range(len(bits_sent))))    
    elif select == '2':
        graph_subplots(tbits, times_res, title + " - TB Bits per TTI", "TB Bits", list(range(len(tbits))))
        graph_subplots(resources, times_res, title + " - Resources per TTI", "PRB", list(range(len(tbits))))
    elif select == '3':
        graph_subplots(bits, times_delay, title + " - Bits in Queue per TTI", "Bits", list(range(len(bits))))
        graph_subplots(pkts, times_delay, title + " - Packets in Queue per TTI", "Packets", list(range(len(bits))))
        graph_subplots(delay, times_delay, title + " - Average Packet Delay in Queue per TTI", "ms", list(range(len(bits))))
    elif select == '4':
        graph_dataG(times_res[0], tbits_sum_01, tbits_sum_23, title + " - Sum of TB Bits per TTI per Group", "TB Bits")
        graph_dataG(times_res[0], resources_sum_01, resources_sum_23, title + " - Sum of Resources per TTI per Group", "PRB")
    elif select == '5':
        graph_dataG(times_delay[0], bits_sum_01_queue, bits_sum_23_queue, title + " - Sum of Bits in Queue per TTI per Group", "Bits")
        graph_dataG(times_delay[0], pkts_sum_01, pkts_sum_23, title + " - Sum of Packets in Queue per TTI per Group", "Packets")
        graph_dataG(times_delay[0], delay_sum_01, delay_sum_23, title + " - Sum of Average Packet Delay in Queue per TTI per Group", "ms")

def main(args):
    conf = load_config()

    if args.set_basestation:
        set_basestation(conf, *args.set_basestation)
    if args.set_uegroup:
        set_uegroup(conf, *args.set_uegroup)
    if args.set_resources:
        set_resources(conf, *args.set_resources)
    if args.set_slice:
        set_slice(conf, *args.set_slice)
    if args.run_simulation:
        run_simulation(conf, args.run_simulation)
    if args.view_config:
        view_config(conf)

    if args.process_data:
        data = process_data(args.process_data)
        print(data)
    if args.graph_data:
        data = process_data(args.graph_data[0])
        graph_data(data, args.graph_data[1], 'Value')
    if args.process_and_graph_data:
        process_and_graph_data(args.process_and_graph_data[0], args.process_and_graph_data[1], args.process_and_graph_data[2])



    save_config(conf)

def pywisim(command):

    #print("DEBUG0")
    parser = argparse.ArgumentParser(description='5G Python Wireless Simulator Command Line Interface')
    
    parser.add_argument('--set-basestation', nargs=12, metavar=('mimo', 'num_slices', 'band', 'name', 'rMCS', 'longcp', 'ul', 'dl', 'time_sim', 'name_sched', 'channel_type', 'channel_mode'), help='Set base station configuration')
    parser.add_argument('--set-uegroup', nargs=7, metavar=('index', 'name', 'par1', 'pkt_size', 'inter_arrival', 'trgen_type', 'num_ues'), help='Set user group configuration')
    parser.add_argument('--set-resources', nargs=6, metavar=('slice_index', 'namedl', 'nameul', 'nresdl', 'nresul', 'sym_slot'), help='Set resources configuration')
    parser.add_argument('--set-slice', nargs=10, metavar=('slice_index', 'slice_name', 'numerology', 'user_groups', 'num_uegroups', 'sym_slot', 'nresdl', 'nresul', 'namedl', 'nameul'), help='Set slice configuration and assign user groups')
    parser.add_argument('--run-simulation', metavar='debug', help='Run the simulation')
    parser.add_argument('--view-config', action='store_true', help='View the current configuration')

    parser.add_argument('--process-data', metavar='file', help='Process simulation data file')
    parser.add_argument('--graph-data', nargs=2, metavar=('file', 'title'), help='Graph data from simulation result file')
    parser.add_argument('--process-and-graph-data', nargs=3, metavar=('selection','subselection', 'title'), help='Process and graph data from simulation result files')


    try:
        args = parser.parse_args(command)
        main(args)
    except SystemExit as e:
        if e.code != 0:
            print(f"Error: {e}")