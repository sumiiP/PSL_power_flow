from function import *

def main() :

    # Define dictionary for bus and line data
    INF_DICT_BUS = {"path": "DATA\Bus.txt", "headers": ["ID", "Bus-type", "P_g", "Q_g", "P_L", "Q_L", "|Vi|", "δi"] }
    INF_DICT_LINE = {"path": "DATA\Line.txt", "headers": ["ID_from", "ID_to", "R", "X", "B"] }
    
    # Read data files (DataFrame form)
    BUS_DATA = read_data_file(INF_DICT_BUS.get('path'), INF_DICT_BUS.get('headers'))
    LINE_DATA = read_data_file(INF_DICT_LINE.get('path'), INF_DICT_LINE.get('headers'))

    # Calculate P,Q net injection power data
    P_net, Q_net = get_net_injection_power(
        BUS_DATA["P_g"], BUS_DATA["Q_g"], BUS_DATA["P_L"], BUS_DATA["Q_L"]
    )
        # BUS_DATA.info()
        # LINE_DATA.info()
    
    # Calculate Admittance matrix
    Y_bus = calc_admittance_matrix(BUS_DATA, LINE_DATA)

    extract_bus_types(BUS_DATA)

    
if __name__ == "__main__":
    main()