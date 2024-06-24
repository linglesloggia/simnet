# -*- coding: utf-8 -*-

def loadModTable():
    """MCS table 2 (5.1.3.1-2) from 3GPP TS 38.214"""
    modTable = []
    modTable.append(
        {
            "spctEff": 0.2344,
            "bitsPerSymb": 2,
            "codeRate": 0.1171875,
            "mcsi": 0,
            "mod": "BPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 0.377,
            "bitsPerSymb": 2,
            "codeRate": 0.1884765625,
            "mcsi": 1,
            "mod": "BPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 0.6016,
            "bitsPerSymb": 2,
            "codeRate": 0.30078125,
            "mcsi": 2,
            "mod": "BPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 0.877,
            "bitsPerSymb": 2,
            "codeRate": 0.4384765625,
            "mcsi": 3,
            "mod": "BPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 1.1758,
            "bitsPerSymb": 2,
            "codeRate": 0.587890625,
            "mcsi": 4,
            "mod": "BPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 1.4766,
            "bitsPerSymb": 4,
            "codeRate": 0.369140625,
            "mcsi": 5,
            "mod": "QPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 1.6953,
            "bitsPerSymb": 4,
            "codeRate": 0.423828125,
            "mcsi": 6,
            "mod": "QPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 1.9141,
            "bitsPerSymb": 4,
            "codeRate": 0.478515625,
            "mcsi": 7,
            "mod": "QPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 2.1602,
            "bitsPerSymb": 4,
            "codeRate": 0.5400390625,
            "mcsi": 8,
            "mod": "QPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 2.4063,
            "bitsPerSymb": 4,
            "codeRate": 0.6015625,
            "mcsi": 9,
            "mod": "QPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 2.5703,
            "bitsPerSymb": 4,
            "codeRate": 0.642578125,
            "mcsi": 10,
            "mod": "QPSK",
        }
    )
    modTable.append(
        {
            "spctEff": 2.7305,
            "bitsPerSymb": 6,
            "codeRate": 0.455078125,
            "mcsi": 11,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 3.0293,
            "bitsPerSymb": 6,
            "codeRate": 0.5048828125,
            "mcsi": 12,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 3.3223,
            "bitsPerSymb": 6,
            "codeRate": 0.5537109375,
            "mcsi": 13,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 3.6094,
            "bitsPerSymb": 6,
            "codeRate": 0.6015625,
            "mcsi": 14,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 3.9023,
            "bitsPerSymb": 6,
            "codeRate": 0.650390625,
            "mcsi": 15,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 4.2129,
            "bitsPerSymb": 6,
            "codeRate": 0.7021484375,
            "mcsi": 16,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 4.5234,
            "bitsPerSymb": 6,
            "codeRate": 0.75390625,
            "mcsi": 17,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 4.8164,
            "bitsPerSymb": 6,
            "codeRate": 0.802734375,
            "mcsi": 18,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 5.1152,
            "bitsPerSymb": 6,
            "codeRate": 0.8525390625,
            "mcsi": 19,
            "mod": "64QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 5.332,
            "bitsPerSymb": 8,
            "codeRate": 0.66650390625,
            "mcsi": 20,
            "mod": "256QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 5.5547,
            "bitsPerSymb": 8,
            "codeRate": 0.6943359375,
            "mcsi": 21,
            "mod": "256QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 5.8906,
            "bitsPerSymb": 8,
            "codeRate": 0.736328125,
            "mcsi": 22,
            "mod": "256QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 6.2266,
            "bitsPerSymb": 8,
            "codeRate": 0.7783203125,
            "mcsi": 23,
            "mod": "256QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 6.5703,
            "bitsPerSymb": 8,
            "codeRate": 0.8212890625,
            "mcsi": 24,
            "mod": "256QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 6.9141,
            "bitsPerSymb": 8,
            "codeRate": 0.8642578125,
            "mcsi": 25,
            "mod": "256QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 7.1602,
            "bitsPerSymb": 8,
            "codeRate": 0.89501953125,
            "mcsi": 26,
            "mod": "256QAM",
        }
    )
    modTable.append(
        {
            "spctEff": 7.4063,
            "bitsPerSymb": 8,
            "codeRate": 0.92578125,
            "mcsi": 27,
            "mod": "256QAM",
        }
    )
    return modTable

def loadtbsTable():
    tbsTable = [24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 
                    144, 152, 160, 168, 176, 184, 192, 208, 224, 240, 256, 272, 288, 304, 320, 
                    336, 352, 368, 384, 408, 432, 456, 480, 504, 528, 552, 576, 608, 640, 672, 
                    704, 736, 768, 808, 848, 888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 
                    1256, 1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800, 1864, 
                    1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536, 2600, 2664, 2728, 
                    2792, 2856, 2976, 3104, 3240, 3368, 3496, 3624, 3752, 3824]
    return tbsTable

def loadSINR_MCStable(tdd):
    """MCS-SINR allocation table"""
    sinrModTable = []

    if tdd:
        ############### TDD #########################
        sinrModTable.append(1.2)  # MCS 0
        sinrModTable.append(3.9)  # MCS 1
        sinrModTable.append(4.9)  # MCS 2
        sinrModTable.append(7.1)  # MCS 3
        sinrModTable.append(7.8)  # MCS 4
        sinrModTable.append(9.05)  # MCS 5
        sinrModTable.append(10.0)  # MCS 6
        sinrModTable.append(11.1)  # MCS 7
        sinrModTable.append(12.0)  # MCS 8
        sinrModTable.append(13.2)  # MCS 9
        sinrModTable.append(14.0)  # MCS 10
        sinrModTable.append(15.2)  # MCS 11
        sinrModTable.append(16.1)  # MCS 12
        sinrModTable.append(17.2)  # MCS 13
        sinrModTable.append(18.0)  # MCS 14
        sinrModTable.append(19.2)  # MCS 15
        sinrModTable.append(20.0)  # MCS 16
        sinrModTable.append(21.8)  # MCS 17
        sinrModTable.append(22.0)  # MCS 18
        sinrModTable.append(22.5)  # MCS 19
        sinrModTable.append(22.9)  # MCS 20
        sinrModTable.append(24.2)  # MCS 21
        sinrModTable.append(25.0)  # MCS 22
        sinrModTable.append(27.2)  # MCS 23
        sinrModTable.append(28.0)  # MCS 24
        sinrModTable.append(29.2)  # MCS 25
        sinrModTable.append(30.0)  # MCS 26
        sinrModTable.append(100.00)  # MCS 27

    else:
        ############## FDD #########################
        sinrModTable.append(0.0)  # MCS 0
        sinrModTable.append(3.0)  # MCS 1
        sinrModTable.append(5.0)  # MCS 2
        sinrModTable.append(7.0)  # MCS 3
        sinrModTable.append(8.1)  # MCS 4
        sinrModTable.append(9.3)  # MCS 5
        sinrModTable.append(10.5)  # MCS 6
        sinrModTable.append(11.9)  # MCS 7
        sinrModTable.append(12.7)  # MCS 8
        sinrModTable.append(13.4)  # MCS 9
        sinrModTable.append(14.0)  # MCS 10
        sinrModTable.append(15.8)  # MCS 11
        sinrModTable.append(16.8)  # MCS 12
        sinrModTable.append(17.8)  # MCS 13
        sinrModTable.append(18.4)  # MCS 14
        sinrModTable.append(20.1)  # MCS 15
        sinrModTable.append(21.1)  # MCS 16
        sinrModTable.append(22.7)  # MCS 17
        sinrModTable.append(23.6)  # MCS 18
        sinrModTable.append(24.2)  # MCS 19
        sinrModTable.append(24.5)  # MCS 20
        sinrModTable.append(25.6)  # MCS 21
        sinrModTable.append(26.3)  # MCS 22
        sinrModTable.append(28.3)  # MCS 23
        sinrModTable.append(29.3)  # MCS 24
        sinrModTable.append(31.7)  # MCS 25
        sinrModTable.append(35.0)  # MCS 26
        sinrModTable.append(100.00)  # MCS 27

    return sinrModTable



def load_numerology_table():
    """Numerologies from 3GPP TS 38.214"""
    num_table = []
    num_table.append(
        {
            "numerology": 0,
            "sub_carrier_spacing": 15,
            "slot_duration": 1,
        }
    )
    num_table.append(
        {
            "numerology": 1,
            "sub_carrier_spacing": 30,
            "slot_duration": 0.5,
        }
    )
    num_table.append(
        {
            "numerology": 2,
            "sub_carrier_spacing": 60,
            "slot_duration": 0.25,
        }
    )
    num_table.append(
        {
            "numerology": 3,
            "sub_carrier_spacing": 120,
            "slot_duration": 0.125,
        }
    )
    num_table.append(
        {
            "numerology": 4,
            "sub_carrier_spacing": 240,
            "slot_duration": 0.0625,
        }
    )
    num_table.append(
        {
            "numerology": 5,
            "sub_carrier_spacing": 480,
            "slot_duration": 0.03125,
        }
    )
    num_table.append(
        {
            "numerology": 6,
            "sub_carrier_spacing": 960,
            "slot_duration": 0.01565,
        }
    )
    return num_table

def load_bands():
    # [fr,bandwidth,tdd] Only Global bands added
    bands={"n1":[1,60,False],"n3":[1,75,False],"n5":[1,25,False],"n8":[1,35,False],"n31":[1,5,False],"n41":[1,194,True], "n46":[1,775,True],"n47":[1,70,True],"n48":[1,150,True],"n65":[1,90,False],"n77":[1,900,True],"n78":[1,500,True],"n79":[1,600,True],"n90":[1,194,True],"n257":[2,3000,True],"n258":[2,3249.9,True],"n259":[2,4000,True],"n260":[2,3000,True],"n261":[2,850,True],"n262":[2,1000,True]}
    
    return bands
