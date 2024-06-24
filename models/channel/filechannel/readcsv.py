
import csv

def mk_dc_chanstate(f_name):
    dc_usreq = {}

    fp = open(f_name, "r", encoding='utf8')
    reader = csv.reader(fp, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL, \
        lineterminator='\n')

    i = 0
    for row in reader:
        #print("ROW", i, row)
        if row[0] in dc_usreq:
            dc_usreq[row[0]] = dc_usreq[row[0]] + [row[1:]]
        else:
            dc_usreq[row[0]] =  [row[1:] ]
        i += 1
        if i > 10:
            break
    fp.close()
    return dc_usreq


def mk_dc_t_states(usreq, dc_chanstate):
    dc_usreq_state = {}
    if usreq in dc_chanstate:
        for item in dc_chanstate[usreq]:
            t_state = item[0]
            state, pos, vel = item[1].split()
            # convert to float and round to 3 decimals
            t_state = round(float(t_state), 3)
            state = round(float(state), 3)
            pos, vel = round(float(pos), 3), round(float(vel), 3) 
            # add to user equipment dictionary of states
            dc_usreq_state[t_state] = [state, pos, vel]
        return dc_usreq_state
    else:
        print("{}: no channel state")
        return {}

dc_ue = mk_dc_chanstate("snr_4ues_1000_5gUmi.csv")
for key in dc_ue:
    print(key)
    for item in dc_ue[key]:
        print("    ", item)

dc_t_states = mk_dc_t_states("UE-3", dc_ue)
print(dc_t_states)


"""
ue_id = "UE-2"

dc_chan_state = {}
ue_states = dc_ue[ue_id]
for item in ue_states:
    #print("item", item)
    time_t, values = item
    state, xx, yy = values.split(" ")
    [float(state), float(xx), float(yy)]
    time_t = float(time_t)
    dc_chan_state[round(time_t, 5)] = [state, xx, yy]

print("\nChannel states for ", ue_id)
for key in dc_chan_state:
    print("    ", key, dc_chan_state[key])

print("state for t=0.003", dc_chan_state[0.003][0])    
print("state for t=0.023", dc_chan_state[0.023][0])    
"""
