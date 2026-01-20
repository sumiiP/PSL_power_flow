from function import *

def main() :

    # Define dictionary for bus and line data
    INF_DICT_BUS = {"path": "DATA\Bus.txt", "headers": ["ID", "Bus-type", "P_g", "Q_g", "P_L", "Q_L", "|Vi|", "δi"] }
    INF_DICT_LINE = {"path": "DATA\Line.txt", "headers": ["ID_from", "ID_to", "R", "X", "B"] }
    
    # Read data files (DataFrame form)
    BUS_DATA = read_data_file(INF_DICT_BUS.get('path'), INF_DICT_BUS.get('headers'))
    LINE_DATA = read_data_file(INF_DICT_LINE.get('path'), INF_DICT_LINE.get('headers'))
    
    # print("<------------Bus data------------>")
    # print(BUS_DATA)
    # print("<------------Line data------------>")
    # print(LINE_DATA)

    # Calculate P,Q net injection power data
    P_net, Q_net = get_net_injection_power(
        BUS_DATA["P_g"], BUS_DATA["Q_g"], BUS_DATA["P_L"], BUS_DATA["Q_L"]
    )
    
    # BUS_DATA.info()
    # LINE_DATA.info()
    
    calc_admittance_matrix(BUS_DATA, LINE_DATA)
    

    # while True : 
    # # Received MATRIX A data from the user
    #     MATRIX_A = input_matrix()   
    # # print received matrix
    #     print(f"MATRIX_A : {MATRIX_A}")

    # # Received MATRIX B data from the user
    #     MATRIX_B = input_matrix()
    # # print received matrix
    #     print(f"MATRIX_B : {MATRIX_B}")

    # # Determining whether MATRIX operations are possible 
    #     if matrix_judgment(MATRIX_A, MATRIX_B) :
    #         break

    # # matrix calculation
    # MULTIPULATION_CAL_RESULT = matrix_multipulication(MATRIX_A, MATRIX_B)

    # print("----------RESULT----------")
    # print(f"{MATRIX_A}*{MATRIX_B} = {MULTIPULATION_CAL_RESULT}")
    
    
if __name__ == "__main__":
    main()