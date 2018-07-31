#!bin/bash/env python
import numpy as np
import numpy.linalg as linalg


def euclid(target,source):
    result = list()
    for s in source:
            result.append(linalg.norm(target-s))
    _ = zip(xrange(len(result)),result)
    return sorted(_,key=lambda x:x[1])

def cosine(target,source):
    result = list()
    for s in source:
            d = np.dot(target, s)
            denom = linalg.norm(target) * linalg.norm(s)
            if d == 0:
                    result.append(-1)
            else:
                    result.append(d/denom)
    _ = zip(xrange(len(result)),result)
    return sorted(_,key=lambda x:x[1])

def main():
    pass

if __name__ == "__main__":
    main()
