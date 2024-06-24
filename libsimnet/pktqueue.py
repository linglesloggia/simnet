#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# simnet: a very elementary simulator example
#

'''
Simple wireless communications simulator example, packet queue module.
'''

from time import sleep, perf_counter
from random import random
import sys

from libsimnet.libutils import mutex_prt
from libsimnet.libstats import meanvar_upd, meanvar_fin


def init_dc_traf():
    '''A dictionary of traffic counters.

    Defined as a module function to be used in different places.
    @return: a dictionary of traffic values.
    '''
    dc_traf = {}                            # traffic dictionary
    dc_traf["Received"] = [0, 0, 0.0, 0.0]  # packets, bits, mean, m2
    dc_traf["Sent"] = [0, 0, 0.0, 0.0]      # packets, bits, mean, m2
    dc_traf["Lost"] = [0, 0, 0.0, 0.0]      # packets, bits, mean, m2
    dc_traf["Dropped"] = [0, 0, 0.0, 0.0]   # packets, bits, mean, m2
    dc_traf["Transport Blocks"] = [0, 0, 0.0, 0.0]  # count, bits, mean, m2
    dc_traf["Delay"] = 0.0
    return dc_traf


def show_dc_traf(dc_traf, meanvar=True, dc_keys=None):
    '''Shows dictionary of traffic counters.

    Defined as a module function to be used in different places.
    @param dc_traf: dictionary of traffic counters to show.
    @param meanvar: if True shows mean and variance.
    '''
    print("Traffic counters")
    if not dc_keys:
        dc_keys = ["Received", "Sent", "Lost", "Dropped", "Transport Blocks"]
    if meanvar:
        print("    {:19s}  {:13s}  {:13s}  {:10s} {:8s}".\
            format("Traffic", "Packets", "Bits", "Mean", "Variance"))
    else:
        print("    {:19s}  {:13s}  {:13s}".\
            format("Traffic", "Packets", "Bits"))
    for nm in dc_keys:
        nr_pkts, nr_bits, bits_mean, bits_m2 = dc_traf[nm]
        if meanvar:
            bits_var = 0 if bits_m2 == 0 else bits_m2/nr_pkts
            print("    {:16s}  {:10d}  {:10d}  {:13.3f}  {:13.3f}".\
                format(nm, nr_pkts, nr_bits, bits_mean, bits_var))
        else:
            print("    {:16}  {:10d}  {:10d}".\
                format(nm, nr_pkts, nr_bits))
    if dc_traf["Sent"][0] > 0:  # there hare packets transmitted
        delay = dc_traf["Delay"] / dc_traf["Sent"][0]
        print("    Transmission delay:  {:11.3f}".format(delay))
    else:
        print("    Transmission delay: no packets transmitted yet.")
    return



class PacketQueue:
    '''A queue of data packets recording reception and transmission.

    Data packets are received from an upper communications layer or from a traffic generator. When a data packet is received, a timestamp of reception is recorded, and the data packet is added to a list of received packets. When a data packet is transmitted, a timestamp of transmission is recorded, and the packet is moved to a list of transmitted packages. This class keeps counters of packets received and transmitted. 
    Format of item in received and sent packet list::

        [ id_packet, time received, time sent, packet object | number of bits ]

    The data packet may be a pointer to an object of a user defined class, a sequence of bytes or a string, or just the size in bits of the data in the packet. If packet is an object, it is convenient to define a __len__() method to determine its size, as well as a __str__() method for pretty printing.

    Timestamp of reception and transmission of packets may be recorded as the number of simulation time units since the simulation started, or as real time. When time is recorded in simulation intervals, the caller must provide the timestamp.

    A prefix for a unique identifier for each data packet received may be provided.

    Transmitted packages may be discarded to save storage, in which case the transmitted list remains empty. Anyway, a counter records the number of transmitted packets.

    A maximum length may be specified for the list of received packets. When this maximum length is exceeded, received packets are dropped. A counter records the number of dropped packets.

    This class supports two transmission modes, single packet and transport block of several packets: 
        - single packet: a packet is extracted from the received list, and moved to the sent list, or optionally discarded after updating counters.
        - transport block transmission: a transport block comprising several packets is made according to a provided maximum transport block size. A transport block may be successfully sent or lost; when successfully sent, packets are optionally moved to a sent list, or discarded; if transport block is lost, packets are moved to a retransmission list. In the next transmission instance, packets in the retransmission queue are given priority.

    Counters of received, sent, lost, retransmitted, and dropped packets are regularly updated. These counters are kept:
        - received packets.
        - sent packets.
        - dropped packets, because maximum length of queue was exceeded.
        - lost packets; this counter is incremented each time a packet is lost; if the same packet is lost several times, counter is incremented on each loss instance.
    '''

    counter = 0
    '''Object creation counter, for unique object identifier.'''


    def __init__(self, qu_prefix="PQ-", pkt_prefix="Pkt-", max_len=0, \
        keep_pkts=True, last_k=50):
        '''Constructor.

        @param qu_prefix: a prefix for queue identifier, default is "PQ-" followed by a sequential number incremented on each PacketQueue created.
        @param pkt_prefix: a prefix for packet identifiers. A packet identifier is this prefix followed by a sequential number incremented on each packet received. Default value is None, in which case the queue identifier is used as prefix.
        @param max_len: maximum length of received packet list, if exceeded packets are dropped; 0 means no maximum length.
        @param keep_pkts: whether to keep transmitted packets in the list of transmitted packets or discarded; may be True to keep all packets, False to discard all, or an integer to keep the indicated number of packets; defauls to True.
        @param last_k: number of last time units to retain traffic bit counters.
        '''

        PacketQueue.counter += 1
        '''Object creation counter, for unique object identifier.'''
        self.id_object = qu_prefix + str(PacketQueue.counter)
        '''A unique identifier for this object.'''
        self.pkt_prefix = pkt_prefix
        '''Packet prefix, for a unique data packet identifier.'''

        self.max_len = max_len
        '''Maximum length of received list, then drop packets.'''
        self.keep_pkts = keep_pkts  # whether to keep transmitted packets
        '''If True keeps all packets, if False discards all, if an integer keeps this number.'''
        self.last_k = last_k 
        '''Number of last time units to retain traffic bit counters.'''

        self.tb_counter = 0
        '''Transport block creation counter.'''
        self.dc_pend = {}
        '''Transport block dictionary, pending of transmission confirmation.'''
        self.ls_recvd = []
        '''List of received packets.'''
        self.ls_sent = []       # list of transmitted packages
        '''List of successfully sent packages.'''
        self.dc_retrans = {}
        '''Dictionary of packets to retransmit, lost in previous transmissions.'''
        self.dc_traf = init_dc_traf()
        '''Dictionary of traffic counters.'''
        self.last_lst_rec = []
        '''List of [time_t, bits_received, dropped] for the last k simulation time units.'''
        self.last_lst_snt = []
        '''List of [time_t, bits_sent, bits_lost, bits_TBs] for the last k simulation time units.'''

        return
            

    def size_bits(self, data_pkt, bits_per_char=8):
        '''Determines size in bits for data packets, string, or integer.

        Returns size in bits of a packet data in this class, or length of a string multiplied by a number of bits per character, or an integer if an integer is received.
        @param data_pkt: a data packet of this class, a string, or an integer representing the packet size as a number of bits.
        @param bits_per_char: number of bits per character, default 8.
        @return: size in bits.
        '''
        if type(data_pkt) is int:           # data is size in bits
            return data_pkt
        elif type(data_pkt) is str:         # data is a string
            return len(data_pkt) * bits_per_char
        elif type(data_pkt) is list:        # data is a PacketQueue packet
            if type(data_pkt[3]) is int:    # payload is size in bits
                return data_pkt[3]
            elif type(data_pkt[3]) is str:  # payload is a string
                return len(data_pkt[3]) * bits_per_char
            else:   # add for objects with __len__ function, with try exception
                print("ERROR in data packet length")
                return 0
        else:
            print("ERROR in data packet length")
            return 0


    def update_dc_traf(self, key, data_pkt):
        '''Updates traffic dictionary for count, bits, mean and m2.

        @param key: traffic dictionary key.
        @param data_pkt:  a data packet.
        '''
        nr_pkts, nr_bits, nr_mean, nr_m2 = self.dc_traf[key]
        nr_bits_pkt = self.size_bits(data_pkt)
        meanvar_aggr = [nr_pkts, nr_mean, nr_m2]  # existing aggregate
        nr_pkts, nr_mean, nr_m2 = meanvar_upd(meanvar_aggr, nr_bits_pkt)
        self.dc_traf[key] = [nr_pkts, nr_bits+nr_bits_pkt, nr_mean, nr_m2]
        return


    def update_last_k(self, time_t, data_pkt, nm_counter):
        '''Adds time and traffic value to list, keeps only last k time values.

        On repeated executions, assumes new time_t >= present time_t (time_t never decreases).
        @param time_t: present instant time.
        @param data_pkt: a data packet object, a string or a number of bits.
        @param nm_counter: the name of the counter.
        '''
        if self.last_k <= 0:    # do not retain last packets
            return
        if nm_counter == "Received":
            last_lst, ix = self.last_lst_rec, 1
        elif nm_counter == "Dropped":
            last_lst, ix = self.last_lst_rec, 2
        elif nm_counter == "Sent":
            last_lst, ix = self.last_lst_snt, 1
        elif nm_counter == "Lost":
            last_lst, ix = self.last_lst_snt, 2
        elif nm_counter == "TranspBlks":
            last_lst, ix = self.last_lst_snt, 3
        val = self.size_bits(data_pkt)
        if not last_lst or last_lst[-1][0] < time_t:
            # if list is empty or time_t in last item < present time_t
            item = [time_t, 0, 0, 0, 0, 0]
            item[ix] = val
            last_lst += [item]            # add new item for present time_t
        elif last_lst[-1][0] == time_t:
            # present time_t is the same as time of last item in list
            last_lst[-1][ix] += val  # add to counter in existing item
        elif last_lst[-1][0] < time_t:
            # present time_t is greater than time of last item in list
            item = [time_t, 0, 0, 0, 0, 0]
            item[ix] = val
            last_lst += [item]            # add new item for present time_t
        while last_lst[0][0] <= time_t - self.last_k:
            # delete first elements older than time_t - time_k
            last_lst = last_lst[1:]
        # update on last_k lists, delete elements older than time_t - time_k
        if nm_counter in ["Received", "Dropped"]:
            self.last_lst_rec = last_lst
        else:
            self.last_lst_snt = last_lst 
        while self.last_lst_rec and self.last_lst_rec[0][0] <= time_t - self.last_k:
            self.last_lst_rec = self.last_lst_rec[1:]
        while self.last_lst_snt and self.last_lst_snt[0][0] <= time_t - self.last_k:
            self.last_lst_snt = self.last_lst_snt[1:]
        return


    def show_last_k(self, time_sim=0):
        '''Shows list of counters for last k time values.

        @param time_sim: total simulation time; if not given and list of last k times packets sent is not empty, last packet sent time is used, else 0 is shown.
        '''
        #print("time_sim", time_sim)
        if time_sim == 0 and len(self.last_lst_snt) > 0:
            time_sim = self.last_lst_snt[-1][0]  # time of last packet sent
        else:
            time_sim = time_sim
        print("Traffic counters for last {} instant times, total time {}".\
            format(self.last_k, time_sim))
        print("    Received:  {:s}   {:s}    {:s}".\
            format("time_t", "Received", "Dropped"))
        if self.last_lst_rec:
            for it in self.last_lst_rec:
                print("           {:10d} {:10d} {:10}".\
                    format(it[0], it[1], it[2]))
        print("    Sent:      {:s}       {:s}       {:s} {:s}".\
            format("time_t", "Sent", "Lost", "TranspBlks"))
        if self.last_lst_snt:
            for it in self.last_lst_snt:
                print("           {:10d} {:10d} {:10} {:10d}".\
                    format(it[0], it[1], it[2], it[3]))
        return


    def receive(self, data_pkt, t_stamp=-1):
        '''Makes a data packet record and adds to received list.

        Makes a packet record as a list, inserts data and reception timestamp in record, inserts record in received packets list.
        @param data_pkt: a data packet object, a string, or a number of bits.
        @param t_stamp: a timestamp of reception, if -1 real time is
        recorded.
        '''
        # if maximum length exceeded, drop packet
        if self.max_len != 0 and len(self.ls_recvd) >= self.max_len: 
            self.update_dc_traf("Dropped", data_pkt)
            self.update_last_k(t_stamp, data_pkt, "Dropped")
            return
        # otherwise accept packet and add to received list
        self.update_dc_traf("Received", data_pkt)
        self.update_last_k(t_stamp, data_pkt, "Received")
        if t_stamp >= 0:
            time_rec = t_stamp         # timestamp, simulation instant time
        else:
            time_rec = perf_counter()  # timestamp, real time
        id_pkt = self.id_object + ":" + self.pkt_prefix + \
            str(self.dc_traf["Received"][0])     # unique packet identifier
        self.ls_recvd += [ [id_pkt, time_rec, 0.0, data_pkt] ]  # add record to list
        return


    def mk_trblk(self, tb_size, time_t, tb_prefix="TB-"):
        '''Returns a transport block of certain size, adds to pending list.

        A transport block is a transport block identifier followed by a list of packets. 
        @param tb_size: size of transport block to build.
        @param time_t: present instant time.
        @param tb_prefix: a prefix for transport block identifier.
        @return: a transport block, or None if received transport block size does not allow to include a packet.
        '''
        ls_pkts = []     # list of packet ids included in the transport block
        tb_size_ini = tb_size
        # first insert from retransmission queue
        for id_pkt in list(self.dc_retrans):
            pkt_to_send = self.dc_retrans[id_pkt]
            if self.size_bits(pkt_to_send) <= tb_size:
                ls_pkts += [pkt_to_send]       # add to packets to send
                tb_size -= self.size_bits(pkt_to_send)
            else:
                break
        # extract packets to send from retransmission
        for pkt in ls_pkts:
            id_pkt = pkt[0]
            if id_pkt in self.dc_retrans:
                del self.dc_retrans[id_pkt]    # extract from retransmission
        # then insert from received data packet list
        while tb_size > 0:
            if not self.ls_recvd:               # no packets in list
                break
            pkt_to_send = self.ls_recvd[0]
            if self.size_bits(pkt_to_send) <= tb_size:
                ls_pkts += [pkt_to_send]        # add to packets to send
                self.ls_recvd = self.ls_recvd[1:]   # extract from received list
                tb_size -= self.size_bits(pkt_to_send)
            else:
                break
        # make transport block
        if ls_pkts:
            tr_blk_id = tb_prefix + str(self.tb_counter)
            self.tb_counter += 1
            tr_blk = [tr_blk_id] + ls_pkts
            self.dc_pend[tr_blk_id] = ls_pkts
            self.update_dc_traf("Transport Blocks", tb_size_ini)
            self.update_last_k(time_t, tb_size_ini, "TranspBlks")
            return tr_blk
        else:
            return None


    def send_tb(self, tr_blk_id, action, t_stamp):
        '''Transport block sent or lost, move to sent or retransmit.

        If the transport block was successfully sent, inserts sent time timestamp and moves packets to sent list; if the transport block was lost, insert packets in retransmission dictionary, if they are not already there.
        @param tr_blk_id: transport block identifier.
        @param action: may be "Lost" or "Sent".
        @param t_stamp: a timestamp of successful transmission.
        '''

        ls_pkts = self.dc_pend[tr_blk_id]
        for data_pkt in ls_pkts:
            id_pkt = data_pkt[0]
            if action == "Sent":        # transport block was successfully sent
                # record time sent
                if t_stamp >= 0:                    # simulation intervals
                    data_pkt[2] = t_stamp           # send timestamp
                else:
                    data_pkt[2] = perf_counter()    # send timestamp
                # update counters; packets sent, bits sent, sent delay
                self.update_dc_traf("Sent", data_pkt)
                self.update_last_k(t_stamp, data_pkt, "Sent")
                self.dc_traf["Delay"] += data_pkt[2] - data_pkt[1]
                # optionally move to sent packets list
                if type(self.keep_pkts) is int:  # number to keep in history
                    self.ls_sent += [data_pkt]   # add to sent packets queue
                    if len(self.ls_sent) > self.keep_pkts:
                        self.ls_sent = self.ls_sent[1:]
                elif self.keep_pkts:             # all packets kept
                    self.ls_sent += [data_pkt]   # add to sent packets queue
                else:
                    pass                         #no history kept
                # if packet in retransmission, extract
                if id_pkt in self.dc_retrans:
                    del self.dc_retrans[id_pkt] 
            elif action == "Lost":      # transport block was lost
                self.update_dc_traf("Lost", data_pkt)
                self.update_last_k(t_stamp, data_pkt, "Lost")
                if id_pkt in self.dc_retrans:
                    pass                            # already in retransmission
                else:
                    self.dc_retrans[id_pkt] = data_pkt   # add for retransmission
        del self.dc_pend[tr_blk_id]             # extract TB from pending
        return


    def send_pkt(self, t_stamp=-1):
        '''Transmission of packet, extracts from received, adds to sent list.

        Simulates direct transmission of a packet, extracting from received list, inserting time sent timestamp, and optionally moving packet to sent packets list.
        @param t_stamp: a timestamp of reception, if None real time is recorded.
        @return: packet to send, or None if there are no packets to send.
        '''

        if self.ls_recvd:                       # list of received packets not void
            pkt_to_send = self.ls_recvd[0]
            self.ls_recvd = self.ls_recvd[1:]   # extract first item
            # record time sent
            if t_stamp >= 0:                        # simulation intervals
                pkt_to_send[2] = t_stamp            # send timestamp
            else:
                pkt_to_send[2] = perf_counter()     # send timestamp
            # optionally move to sent packets list
            if type(self.keep_pkts) is int:  # number to keep in history
                self.ls_sent += [pkt_to_send]   # add to sent packets queue
                if len(self.ls_sent) > self.keep_pkts:
                    self.ls_sent = self.ls_sent[1:]
            elif self.keep_pkts:             # all packets kept
                self.ls_sent += [pkt_to_send]   # add to sent packets queue
            else:
                pass                         # no history kept
            # update counters; packets sent, bits sent, sent delay
            self.update_dc_traf("Sent", pkt_to_send)
            self.update_last_k(t_stamp, pkt_to_send, "Sent")
            self.dc_traf["Delay"] += pkt_to_send[2] - pkt_to_send[1]
        else:
            pass  # no packet to send
            return None


    def get_state(self):
        '''Returns number of packets to transmit, retransmit.

        @return: number of packets to transmit, number of packets to retransmit.
        '''
        pkts_trans = len(self.ls_recvd)
        pkts_retrans = len(self.dc_retrans)
        return pkts_trans, pkts_retrans


    def show_packet(self, packet):
        '''Show record for a packet, its metadata and the packet.

        @param packet: a packet object, a string, or a number of bits.
        '''
        id_pkt, time_rec, time_sent, pkt = packet
        print("    {:15s}  {:10.3f}  {:10.3f}   {:12} {:10d}".\
              format(id_pkt, time_rec, time_sent, pkt, self.size_bits(pkt)) )
        return


    def show_ls_pkts(self, list_type):
        '''Show all packet records in the list, and counters.

        @param list_type: may be "Sent" or "Received".
        '''
        print("{:s} packet list: received {}, sent {}, dropped {} ".\
              format(list_type, self.dc_traf["Received"][0], 
                self.dc_traf["Sent"][0], self.dc_traf["Dropped"][0]) )
        #print("    packet id          received        sent   pkt/bits           bits")
        print("    {:18s} {:15s} {:7s} {:17s} {:s}".format("packet id", \
            "received", "sent", "pkt/bits", "bits"))
        if list_type == "Received":
            for packet in self.ls_recvd:
                self.show_packet(packet)
        elif list_type == "Sent":
            for packet in self.ls_sent:
                self.show_packet(packet)
        return


    def show_trblk(self, tr_blk):
        '''Shows packets in a transport block.

        @param tr_blk: a transport block, a list of packet identifiers.
        '''
        print("Packets in transport block {:s}:".format(tr_blk[0]))
        for pkt in tr_blk[1:]:
            self.show_packet(pkt)
        return


    def show_pending(self):
        '''Show transport blocks in pending queue.
        '''
        print("Pending queue:")
        for tr_blk_id in self.dc_pend.keys():
            print("    {:s}".format(tr_blk_id))
            #print(self.dc_pend)
            for ls_pkt in self.dc_pend[tr_blk_id]:
                print("        {}".format(ls_pkt))
        return


    def show_retrans(self):
        '''Show retransmission dictionary.
        '''
        print("Retransmission dictionary:")
        for (id_pkt, packet) in self.dc_retrans.items():
            print("    {:s}  {}".format(id_pkt, packet)) 
        return


    def get_bits(self):
        '''Returns bit counters of received, sent and lost packets.

        @return: a tuple of bits received, bits sent, and bits to send.
        '''
        bits_to_send = self.dc_traf["Received"][1] - self.dc_traf["Sent"][1]
        return self.dc_traf["Received"][1], self.dc_traf["Sent"][1], bits_to_send


    def show_counters(self):
        '''Shows list indexes and counters in number of packets and bits.

        Uses module function.
        '''
        show_dc_traf(self.dc_traf)
        return


    def __str__(self):
        '''For pretty printing.'''
        msg = "PacketQueue "
        msg += "{:s}, keep {}, max {:d}; rec {:d}, sent {:d}, queue {:d}".\
            format(self.id_object, self.keep_pkts, self.max_len, 
                self.dc_traf["Received"][0], self.dc_traf["Sent"][0],
                len(self.ls_recvd) )
        return msg


    
