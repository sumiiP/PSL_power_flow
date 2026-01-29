import numpy as np
import pandas as pd

# # Calculate net injection power
# def get_net_injection_power(P_g, Q_g, P_L, Q_L) :
#     P_net = P_g - P_L
#     Q_net = Q_g - Q_L
#     # print(type(P_net))
#     return P_net, Q_net

# # New function to get theta and voltage vectors
# def get_theta_voltage_vector(BUS_DATA) :
#     theta_vector = np.radians(BUS_DATA['δi'].values)
#     voltage_vector = BUS_DATA['|Vi|'].values
    
#     return theta_vector, voltage_vector

# def get_admittance_matrix(BUS_DATA, LINE_DATA) :
#     # Initialize Y_bus matrix
#     Y_BUS = np.zeros((BUS_DATA.shape[0], BUS_DATA.shape[0]), dtype=complex)
    
#     for i in range(LINE_DATA.shape[0]) :
#         # Line DATA columns - iloc index : [0:"ID_from", 1:"ID_to", 2:"R", 3:"X", 4:"B"]
#         row = LINE_DATA.iloc[i]
#         # print(f" 첫번째 : {LINE_DATA.iloc[i]['ID_from']}")
#         # print(row.iloc[0] - 1)
#         k = int(LINE_DATA.iloc[i]['ID_from']) - 1  # Adjusting for zero-based index
#         n = int(LINE_DATA.iloc[i]['ID_to']) - 1    # Adjusting for zero-based index
        
#         R = LINE_DATA.iloc[i]['R']
#         X = LINE_DATA.iloc[i]['X']
#         B = LINE_DATA.iloc[i]['B']
#         Z = complex(R, X)
#         Y = 1 / Z
#         B_shunt = complex(0, B / 2)
        
#         # Self-admittance
#         Y_BUS[k, k] += Y + B_shunt
#         Y_BUS[n, n] += Y + B_shunt
#         # Mutual admittance
#         Y_BUS[k, n] -= Y
#         Y_BUS[n, k] -= Y
        
#         # debugging print
#         # Y_bus_df = pd.DataFrame(Y_BUS, index=BUS_DATA['ID'], columns=BUS_DATA['ID'])
#         # print(f"After processing line {i+1} (from bus {k+1} to bus {n+1}):")
#         # print(Y_bus_df)
#     return Y_BUS

# def get_conductance_susceptance_matrices(Y_BUS) :
#     G_MATRIX = Y_BUS.real
#     B_MATRIX = Y_BUS.imag
    
#     # print(G_MATRIX, B_MATRIX)
#     return G_MATRIX, B_MATRIX

# def extract_bus_types(BUS_DATA) :
#     bus_types = BUS_DATA['Bus-type'].unique()
#     BUS_TYPE_DICT = {}
    
#     for b_type in bus_types :
#         BUS_TYPE_DICT[b_type] = BUS_DATA[BUS_DATA['Bus-type'] == b_type]['ID'].tolist()
    
#     return BUS_TYPE_DICT

# # expressed in imaginary coordinates
# def get_calculated_powers_imaginary(pq_bus_ids, Y_BUS, P_net, Q_net, voltage_vector, theta_vector) :
#     RESULT = []
#     for bus_id in pq_bus_ids :
#         i = bus_id - 1  # Adjusting for zero-based index
#         P_calc = 0
#         Q_calc = 0
        
#         # off-diagonal element contribution - mutual admittance
#         for j in range(len(voltage_vector)) :
#             if i != j :
#                 P_calc += voltage_vector[i] * voltage_vector[j] * (Y_BUS[i, j].real * np.cos(theta_vector[i] - theta_vector[j]) + Y_BUS[i, j].imag * np.sin(theta_vector[i] - theta_vector[j]))
#                 Q_calc += voltage_vector[i] * voltage_vector[j] * (Y_BUS[i, j].real * np.sin(theta_vector[i] - theta_vector[j]) - Y_BUS[i, j].imag * np.cos(theta_vector[i] - theta_vector[j]))
#         # diagonlal element contribution - self admittance
#         P_calc += voltage_vector[i]**2 * Y_BUS[i, i].real
#         Q_calc += voltage_vector[i]**2 * (-Y_BUS[i, i].imag)
        
#         # print(f"Bus ID: {bus_id}, Calculated P: {P_calc:.4f}, Given P_net: {P_net.iloc[i]:.4f}, Calculated Q: {Q_calc:.4f}, Given Q_net: {Q_net.iloc[i]:.4f}")
#         RESULT.append(({'bus_id': bus_id, 'P_calc': P_calc, 'Q_calc': Q_calc}))
#     print(RESULT)
    
#     return RESULT
# # expressed in rectangular coordinates
# def get_calculated_powers_rectangular(pq_bus_ids, G_MATRIX, B_MATRIX, P_net, Q_net, voltage_vector, theta_vector) :
#     RESULT = []
#     for bus_id in pq_bus_ids :
#         i = bus_id - 1  # Adjusting for zero-based index
#         P_calc = 0
#         Q_calc = 0
        
#         # off-diagonal element contribution - mutual admittance
#         for j in range(len(voltage_vector)) :
#             if i != j :
#                 P_calc += voltage_vector[i] * voltage_vector[j] * (G_MATRIX[i, j] * np.cos(theta_vector[i] - theta_vector[j]) + B_MATRIX[i, j] * np.sin(theta_vector[i] - theta_vector[j]))
#                 Q_calc += voltage_vector[i] * voltage_vector[j] * (G_MATRIX[i, j] * np.sin(theta_vector[i] - theta_vector[j]) - B_MATRIX[i, j] * np.cos(theta_vector[i] - theta_vector[j]))
#                 # print(f"G_MATRIX[{i},{j}] : {G_MATRIX[i,j]}, B_MATRIX[{i},{j}] : {B_MATRIX[i,j]}")
#         # diagonlal element contribution - self admittance
#         P_calc += voltage_vector[i]**2 * G_MATRIX[i, i]
#         Q_calc += voltage_vector[i]**2 * (-B_MATRIX[i, i])
#         # print(f"G_MATRIX row {i} data: {G_MATRIX[i, :]}") 
#         # print(f"Bus ID: {bus_id}, Calculated P: {P_calc:.4f}, Given P_net: {P_net.iloc[i]:.4f}, Calculated Q: {Q_calc:.4f}, Given Q_net: {Q_net.iloc[i]:.4f}")
#         RESULT.append(({'bus_id': bus_id, 'P_calc': P_calc, 'Q_calc': Q_calc}))
#     print(RESULT)
#     return RESULT

# def input_matrix() :
#     while True : 
#         try : 
#             # Received data for matrix A from the user
#             MAT_VALUE = input("Enter matrix in a single line the left (ex. 1,2,3,...) :")
#             MAT_ROW = input("Enter the 'column' form of matrix A (ex. 5): ")
            
#             # datatype preprocessing (string to int)
#             number_list_A = [float(x.strip()) for x in MAT_VALUE.split(',')]
#                 # [TBD] edit by deleting spaces!!!
            
#             # matrix form reshape
#             MATRIX = np.array(number_list_A).reshape(-1, int(MAT_ROW))
            
#             #error except process : TBD
#             return MATRIX
        
#         except ValueError :
#             print("[ERROR] : Please enter only numbers and follow the format. try agiain")
#             return None
#         except Exception as e : 
#             print(f"[ERROR]_{e} : Please enter only numbers and follow the format. try agiain")
#             return None
        
# def matrix_judgment (mat_a, mat_b) :
#     if mat_a.shape[1] == mat_b.shape[0]:
#         return True
#     else : 
#         print(f"Not possible to multipulication: A_column({mat_a.shape[1]}) != B_row({mat_b.shape[0]})")
#         return False

# def matrix_multipulication(mat_a, mat_b) :
#     RESULT = np.matmul(mat_a, mat_b)
    
#     return RESULT