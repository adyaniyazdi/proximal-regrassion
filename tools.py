import numpy as np
import random



def banner(msg):
    print("=" * 75)
    print(msg)
    print("=" * 65)

def minimize(f, low_lim, up_lim, accuracy):

    while (up_lim - low_lim > accuracy):
        difference = up_lim - low_lim
        l1 = low_lim + (difference/3.0)
        l2 = low_lim + (2*difference/3.0)
        # print("l1:", l1, "l2:", l2)
        if (f(l1) > f(l2)):
            low_lim = l1
        else:
            up_lim = l2
    return (up_lim + low_lim)/2