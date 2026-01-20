import numpy as np
import pandas as pd

def read_data_file(file_path, headers) :
    try : 
        RESULT_MATRIX = pd.read_csv(file_path, sep='\t', header=None)
        # [TBD] datatype 확인 및 변환 필요시 처리

        # row,columns shape 확인
        rows, cols = RESULT_MATRIX.shape
        
        # header 추가
        if len(headers) == int(cols) :
            RESULT_MATRIX.columns = headers
            return RESULT_MATRIX
        else : 
            print("Header information error.")
            return None
        
    except Exception as e : 
        print(f"An error occurred while reading the file : {e}")
        return None    

def get_net_injection_power(P_g, Q_g, P_L, Q_L) :
    P_net = P_g - P_L
    Q_net = Q_g - Q_L
    # print(type(P_net))
    return P_net, Q_net
    
def calc_admittance_matrix(BUS_DATA, LINE_DATA) :
    # Initialize Y_bus matrix
    Y_bus = np.zeros((BUS_DATA.shape[0], BUS_DATA.shape[0]), dtype=complex)
    
    # [TBD] BUS_DATA에서 BUS TYPE에 따른 처리 필요시 추가 (ex: slack bus 등)
    for i in range(LINE_DATA.shape[0]) :
        k = int(LINE_DATA.iloc[i]['ID_from']) - 1  # Adjusting for zero-based index
        n = int(LINE_DATA.iloc[i]['ID_to']) - 1    # Adjusting for zero-based index
        
        R = LINE_DATA.iloc[i]['R']
        X = LINE_DATA.iloc[i]['X']
        B = LINE_DATA.iloc[i]['B']
        Z = complex(R, X)
        Y = 1 / Z
        B_shunt = complex(0, B / 2)
        
        # Self-admittance
        Y_bus[k, k] += Y + B_shunt
        Y_bus[n, n] += Y + B_shunt
        # Mutual admittance
        Y_bus[k, n] -= Y
        Y_bus[n, k] -= Y
        
        # debugging print
        # Y_bus_df = pd.DataFrame(Y_bus, index=BUS_DATA['ID'], columns=BUS_DATA['ID'])
        # print(f"After processing line {i+1} (from bus {k+1} to bus {n+1}):")
        # print(Y_bus_df)

    return Y_bus

def extract_bus_types(BUS_DATA) :
    bus_types = BUS_DATA['Bus-type'].unique()
    bus_type_dict = {}
    
    for b_type in bus_types :
        bus_type_dict[b_type] = BUS_DATA[BUS_DATA['Bus-type'] == b_type]['ID'].tolist()
    
    return bus_type_dict

#def calc_voltage_magnitude_angle(///)

def input_matrix() :
    while True : 
        try : 
            # Received data for matrix A from the user
            MAT_VALUE = input("Enter matrix in a single line the left (ex. 1,2,3,...) :")
            MAT_ROW = input("Enter the 'column' form of matrix A (ex. 5): ")
            
            # datatype preprocessing (string to int)
            number_list_A = [float(x.strip()) for x in MAT_VALUE.split(',')]
                # [TBD] edit by deleting spaces!!!
            
            # matrix form reshape
            MATRIX = np.array(number_list_A).reshape(-1, int(MAT_ROW))
            
            #error except process : TBD
            return MATRIX
        
        except ValueError :
            print("[ERROR] : Please enter only numbers and follow the format. try agiain")
            return None
        except Exception as e : 
            print(f"[ERROR]_{e} : Please enter only numbers and follow the format. try agiain")
            return None
        
def matrix_judgment (mat_a, mat_b) :
    if mat_a.shape[1] == mat_b.shape[0]:
        return True
    else : 
        print(f"Not possible to multipulication: A_column({mat_a.shape[1]}) != B_row({mat_b.shape[0]})")
        return False

def matrix_multipulication(mat_a, mat_b) :
    RESULT = np.matmul(mat_a, mat_b)
    
    return RESULT