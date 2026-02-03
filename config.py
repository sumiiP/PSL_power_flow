# File path specification
BUS_FILE_PATH = "DATA\Bus.txt"
LINE_FILE_PATH = "DATA\Line.txt"

# Each file header specification
BUS_FILE_HEADER_LIST = ["ID", "Bus-type", "P_g", "Q_g", "P_L", "Q_L", "|Vi|", "δi"]
LINE_FILE_HEADER_LIST = ["ID_from", "ID_to", "R", "X", "B"]

# Newton-Raphson Iteration config
MAX_ITER = 20
TOLERANCE = 1e-6

# Power Flow Calculation config (Rectangular Coordinate)
LOAD_FACTOR = 0.01
DAMPING_FACTOR = 0.5

# Tolerance for determining the law of conservation of energy (Gen = Load + Loss)
ENERGY_BALANCE_TOLERANCE = 1e-6