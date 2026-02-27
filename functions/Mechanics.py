import numpy as np


def rotm(th,ux,uy,uz):
    r = np.array((
        ( ux*ux*(1-np.cos(th))+np.cos(th),     ux*uy*(1-np.cos(th))-uz*np.sin(th),     ux*uz*(1-np.cos(th))+uy*np.sin(th) ),
        ( ux*uy*(1-np.cos(th))+uz*np.sin(th),  uy*uy*(1-np.cos(th))+np.cos(th),        uy*uz*(1-np.cos(th))-ux*np.sin(th) ),
        ( ux*uz*(1-np.cos(th))-uy*np.sin(th),  uy*uz*(1-np.cos(th))+ux*np.sin(th),     uz*uz*(1-np.cos(th))+np.cos(th)    )
        )) 
    return r

#(1) Pitch of thp around PitchAx, and rotation of Yaw and Roll axis:
def rotm1(thp,pitchax,rollax,yawax):    
    r1 = rotm(np.pi/2-thp,pitchax[0],pitchax[1],pitchax[2])
    rollax2 = r1.dot(rollax)
    yawax2  = r1.dot(yawax)
    return r1, rollax2, yawax2

#(2) Yaw of thy around yawax2, and rotation of Roll axis:
def rotm2(thy,rollax2,yawax2):    
    r2 = rotm(thy,yawax2[0],yawax2[1],yawax2[2])
    rollax3 = r2.dot(rollax2)    
    return r2, rollax3

#(3) Roll of thr around Rollax3:
def rotm3(thr,rollax3):    
    r3 = rotm(thr,rollax3[0],rollax3[1],rollax3[2])
    return r3



def kirot(thp,thy,thr,n0, pitchax,rollax,yawax):
    #note: it seems like in Alberto's tool the roll and yaw are not transformed by subsequent rotations. For comparison, Ileave this out too.

    r = rotm1(thp,pitchax,rollax,yawax)[0]@rotm3(thr,rollax)@rotm2(thy,rollax,yawax)[0]
    
    

    r1, rollax2, yawax2 = rotm1(thp,pitchax,rollax,yawax)
    rollax2=rollax
    yawax2=yawax
    r2, rollax3 = rotm2(thy,rollax2,yawax2)
    rollax3=rollax
    r3 = rotm3(thr,rollax3)    

    return r3.dot(r2.dot(r1.dot(n0)))#r.dot(n0)#