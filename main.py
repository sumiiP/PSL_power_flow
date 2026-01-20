from function import *

def main() :

    # Define dictionary for bus and line data
    INF_DICT_BUS = {"path": "DATA\Bus.txt", "headers": ["ID", "Bus-type", "P_g", "Q_g", "P_L", "Q_L", "|Vi|", "δi"] }
    INF_DICT_LINE = {"path": "DATA\Line.txt", "headers": ["ID_from", "ID_to", "R", "X", "B"] }
    
    BUS_VAL = read_data_file(INF_DICT_BUS.get('path'), INF_DICT_BUS.get('headers'))
    LINE_VAL = read_data_file(INF_DICT_LINE.get('path'), INF_DICT_LINE.get('headers'))
    
    print("<------------Bus data------------>")
    print(BUS_VAL)
    print("<------------Line data------------>")
    print(LINE_VAL)
    
    
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