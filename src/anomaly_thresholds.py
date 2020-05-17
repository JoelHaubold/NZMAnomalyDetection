
from enum import Enum


class AThreshold(Enum):
    phase_dif = 7.34
    SeasDif = 5.87
    SeasDifALt = 3.23
    StationDif = 8.772
    TimeDif = 179
    TrafoDif = 0.1 # this low becuase per second
    TrafoDifAlt = 0.15


if __name__ == '__main__':
    x = AThreshold.phase_dif
    print(x.value)
    print(x.name)
