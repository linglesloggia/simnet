# IN PROGRESS


# test UserEquipment pos_move()
from libsimnet.usernode import UserEquipment

ue = UserEquipment("NN", v_pos=[0,0,0], v_vel=[1,2,3])
print(ue)

ue.pos_move(10)
print(ue)

ue.pos_move(20)
print(ue)

ue.pos_move(30, [4,5,6])
print(ue)



