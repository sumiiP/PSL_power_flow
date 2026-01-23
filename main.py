from function import *

def main() :

    # Define dictionary for bus and line data
    INF_DICT_BUS = {"path": "DATA\Bus.txt", "headers": ["ID", "Bus-type", "P_g", "Q_g", "P_L", "Q_L", "|Vi|", "δi"] }
    INF_DICT_LINE = {"path": "DATA\Line.txt", "headers": ["ID_from", "ID_to", "R", "X", "B"] }
    
    # Read data files (DataFrame form)
    BUS_DATA = read_data_file(INF_DICT_BUS.get('path'), INF_DICT_BUS.get('headers'))
    LINE_DATA = read_data_file(INF_DICT_LINE.get('path'), INF_DICT_LINE.get('headers'))
        # BUS_DATA.info()
        # LINE_DATA.info()
        
    # Calculate P,Q net injection power data
    P_net, Q_net = get_net_injection_power(
        BUS_DATA["P_g"], BUS_DATA["Q_g"], BUS_DATA["P_L"], BUS_DATA["Q_L"]
    )
    theta_vector, voltage_vector = get_theta_voltage_vector(BUS_DATA)
    
        #BUS_DATA['P_net'] = P_net
        #BUS_DATA['Q_net'] = Q_net
    
    # Calculate Admittance matrix
    Y_BUS = get_admittance_matrix(BUS_DATA, LINE_DATA)
    # Create G and B matrices
    G_MATRIX, B_MATRIX = get_conductance_susceptance_matrices(Y_BUS)

    # Extract bus types
    BUS_TYPE_DICT = extract_bus_types(BUS_DATA)
    
    #Calculate powers based on bus types
    if BUS_TYPE_DICT.get(3) : # PQ Bus exists
        get_calculated_powers_imaginary(BUS_TYPE_DICT.get(3), Y_BUS, P_net, Q_net, voltage_vector, theta_vector)
        get_calculated_powers_rectangular(BUS_TYPE_DICT.get(3), G_MATRIX, B_MATRIX, P_net, Q_net, voltage_vector, theta_vector)
    
    elif BUS_TYPE_DICT.get(2) : # PV Bus exists
        pass
    
    else : # Only Slack Bus exists
        pass    
    
if __name__ == "__main__":
    main()