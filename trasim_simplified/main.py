import timeit

from core.lane import Lane
from core.constant import CFM

def run():
    lane = Lane(length=10000)
    lane.addDriverTool(carNum=100,
                       fRule=CFM.GIPPS)

    for i in range(5000):
        lane.step()
        # print(lane.getDriverPos())


if __name__ == '__main__':
    print(globals())
    print(timeit.timeit('run()',number=1, globals=globals()))
