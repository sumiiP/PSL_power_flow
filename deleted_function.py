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