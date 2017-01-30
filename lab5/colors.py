import random

COLORS = {
    'RED':    (255, 0,   0),
    'GREEN':  (0,   255, 0),
    'BLUE':   (0,   0,   255),
    'DUNNO1': (0,   255, 255),
    'DUNNO2': (255, 0,   255),
    'DUNNO3': (255, 255, 0),
}

def getRandomColor():
    return random.choice(COLORS.values())
