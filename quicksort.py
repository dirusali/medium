from functools import lru_cache
from numba import jit

def listToTuple(function):
    def wrapper(*args):
        args = [tuple(x) if type(x) == list else x for x in args]
        result = function(*args)
        result = tuple(result) if type(result) == list else result
        return result
    return wrapper

import random
l = [random.randint(1,1000000) for i in range(1,1000000)]

def ordena(l):
    p = l[0]
    u = l[len(l)-2]
    pu = l[len(l)-3]
    si = []
    sd = []
    final = []
    pivot = (p+u+pu)//3
    start = 0
    end = 0
    print(pivot)
    for i in l:      
        if i < pivot:
            si.append(i)
        else:
            sd.append(i)
    fd = ordenad(sd)
    fi = ordenad(si)
    final = fi + fd
    return final
    
@jit
def ordenad(sd):
    fd = []
    n = len(sd)//2 + 1
    p = sd[0]
    print(n)
    fd.append(p)
    sd =sd[1:]
    for i in sd:
        for x in range(n+1):
            if i >= fd[x]:
                if i >= fd[len(fd)-1]:
                    fd.append(i)
                    break
                elif i < fd[x]:
                    fd.insert(x,i)
                    continue
            elif i < fd[x]:
                fd.insert(x,i)
                break
    print('------FIN------')
    return fd  
    
    
