import pandas as pd
import numpy as np

class PowerFlowLoader : 
    def __init__(self, y_bus, bus_df, id_info, line_df) : 
        self.Y = y_bus # (33, 33)
        self.df = bus_df
        
        # mismatch judgment index
        self.p_idx = [int(i) - 1 for i in id_info['p_id']] # 1 ~ 32
        self.q_idx = [int(i) - 1 for i in id_info['q_id']] # 1 ~ 32
        
        # initial setting
        self.e = bus_df['e'].values.astype(float) # (33, )
        self.f = bus_df['f'].values.astype(float) # (33, )
        self.v = bus_df['|Vi|'].values.astype(float) # (33, )
        self.delta = np.radians(bus_df['δi'].values.astype(float)) # (33, )

        self.p_net = bus_df['P_net'].values # (33, )
        self.q_net = bus_df['Q_net'].values # (33, )
        self.p_calc = None
        self.q_calc = None

        # Used to calculate final results
        self.line_df = line_df
        
    def _calculate_current_power_polar(self) :
        # V = |V|*(cosδ + jsinδ)
        V_complex = self.v * (np.cos(self.delta) + 1j * np.sin(self.delta)) # (33, )
        # I = YV
        I_complex = self.Y @ V_complex # (33, )
        # S = V * conj(I) = P + jQ
        S_complex = V_complex * np.conj(I_complex) # (33, )
        
        self.p_calc = S_complex.real
        self.q_calc = S_complex.imag
        
    def _calculate_current_power_rectangular(self) :
        # V = e + jf
        V = self.e + 1j * self.f # (33, )
        # I = YV
        I = self.Y @ V # (33, )
        # S = V * conj(I) = P + jQ
        S = V * np.conj(I) # (33, )
        
        self.p_calc = S.real
        self.q_calc = S.imag
        
        # return self.p_calc, self.q_calc
    
    def _make_mismatch_vector(self) : # calculate to delta y
        # self._calculate_current_power_rectangular
        self._calculate_current_power_polar()
        
        # Mismatch vector calculation
        # delta_p and delta_q do not include slack bus !!!
        delta_p = self.p_net[self.p_idx] - self.p_calc[self.p_idx] # ΔP = P_net - P_calc
        delta_q =  self.q_net[self.q_idx] - self.q_calc[self.q_idx] # ΔQ = Q_net - Q_calc
        
        mismatch_vector = np.concatenate([delta_p, delta_q]) # (64, )
        
        return mismatch_vector
    
    def _make_jacobian_polar(self) :
        V = self.v
        delta = self.delta
        G, B = self.Y.real, self.Y.imag
        
        num_bus_reduced = len(V) - 1 # remove slack bus number from entire bus number
        
        H = np.zeros((num_bus_reduced, num_bus_reduced)) # dP/dδ
        N = np.zeros((num_bus_reduced, num_bus_reduced)) # dP/dV
        K = np.zeros((num_bus_reduced, num_bus_reduced)) # dQ/dδ
        L = np.zeros((num_bus_reduced, num_bus_reduced)) # dQ/dV
        
        for i in range(num_bus_reduced) :
            actual_i = i + 1
            Vi = V[actual_i]
            Pi, Qi = self.p_calc[actual_i], self.q_calc[actual_i]
            
            for j in range(num_bus_reduced) :
                actual_j = j + 1
                Vj = V[actual_j]
                ang_ij = delta[actual_i] - delta[actual_j] # delta_i - delta_j
                
                sin_ij = np.sin(ang_ij)
                cos_ij = np.cos(ang_ij)
                Gij, Bij = G[actual_i, actual_j], B[actual_i, actual_j]
                
                if i == j :  # Diagonal elements
                    Vi2 = Vi**2
                    H[i, j] = -Qi - B[actual_i, actual_i] * Vi2
                    N[i, j] =  Pi / Vi + G[actual_i, actual_i] * Vi
                    K[i, j] =  Pi - G[actual_i, actual_i] * Vi2
                    L[i, j] =  Qi / Vi - B[actual_i, actual_i] * Vi
                    
                else :  # Off-diagonal elements
                    ViVj = Vi * Vj
                    H[i, j] =  ViVj * (Gij * sin_ij - Bij * cos_ij)
                    N[i, j] =  Vi   * (Gij * cos_ij + Bij * sin_ij)
                    K[i, j] = -ViVj * (Gij * cos_ij + Bij * sin_ij)
                    L[i, j] =  Vi   * (Gij * sin_ij - Bij * cos_ij)
            
        J = np.block([[H, N], [K, L]])
        
        return J
    
    # Jacobian matrix calculation in rectangular coordinates (but, diverges when the system is large)
    def _make_jacobian_rectangualr(self) :
        e, f = self.e, self.f
        G, B = self.Y.real, self.Y.imag

        # I = YV
        V = e + 1j * f
        I = self.Y @ V
        Ir, Ii = I.real, I.imag

        # Jacobian sub matrices initialization
        num_bus_reduced = len(e) - 1 # remove slack bus number from entire bus number
        
        H = np.zeros((num_bus_reduced, num_bus_reduced)) # dP/df
        N = np.zeros((num_bus_reduced, num_bus_reduced)) # dP/de
        K = np.zeros((num_bus_reduced, num_bus_reduced)) # dQ/df
        L = np.zeros((num_bus_reduced, num_bus_reduced)) # dQ/de

        for i in range(num_bus_reduced) :
            actual_i = i + 1
            
            for j in range(num_bus_reduced) :
                actual_j = j + 1
                
                if i == j : # Diagonal elements
                    # P_calc = e*Ir + f*Ii / Q_calc = f*Ir - e*Ii 
                    H[i, j] = -(e[actual_i]*B[actual_i, actual_i] + f[actual_i]*G[actual_i, actual_i] + Ii[actual_i]) # dP/df
                    N[i, j] = -(e[actual_i]*G[actual_i, actual_i] - f[actual_i]*B[actual_i, actual_i] + Ir[actual_i]) # dP/de
                    K[i, j] = -(e[actual_i]*G[actual_i, actual_i] - f[actual_i]*B[actual_i, actual_i] - Ir[actual_i]) # dQ/df
                    L[i, j] = -(-e[actual_i]*B[actual_i, actual_i] - f[actual_i]*G[actual_i, actual_i] + Ii[actual_i]) # dQ/de
                else : # Off-diagonal elements (i != j)
                    H[i, j] = -(e[actual_i] * B[actual_i, actual_j] + f[actual_i] * G[actual_i, actual_j])
                    N[i, j] = -(e[actual_i] * G[actual_i, actual_j] - f[actual_i] * B[actual_i, actual_j])
                    K[i, j] = -(e[actual_i] * G[actual_i, actual_j] - f[actual_i] * B[actual_i, actual_j])
                    L[i, j] = -(-e[actual_i] * B[actual_i, actual_j] - f[actual_i] * G[actual_i, actual_j])
        
        J = np.block([[H, N], [K, L]])
        
        return J

    def calculate_power(self, MAX_ITER, TOLERANCE, LOAD_FACTOR, DAMPING_FACTOR) :
        # self.p_net *= LOAD_FACTOR
        # self.q_net *= LOAD_FACTOR
        
        # Newton-Raphson Iteration loop
        for iteration in range(MAX_ITER) :
            # 1. Mismatch vector calculation
            mismatch_vector = self._make_mismatch_vector()
                
            # 2. Convergence check
            max_mismatch = np.max(np.abs(mismatch_vector))
            if max_mismatch < TOLERANCE :
                print(f"Converged in {iteration} iterations!! Max mismatch: {max_mismatch}")
                return True
            
            # 3. Jacobian matrix calculation
            # J = self._make_jacobian_rectangualr() # using rectangualr coordinates
            J = self._make_jacobian_polar() # using polar coordinates
                        
            # 4. Solve for state variable updates
            delta_x = np.linalg.solve(J, mismatch_vector)
            
            # n = len(self.e) - 1 # number of non-slack buses (on rectangular)
            n = len(self.v) - 1 # number of non-slack buses (on polar)
            
            # 5. Update state variables
            # self.f[1:] += DAMPING_FACTOR * delta_x[:n] # skip slack bus (on rectangular, f is not used)
            # self.e[1:] += DAMPING_FACTOR * delta_x[n:] # skip slack bus (on rectangular, e is not used)
            self.delta[1:] += delta_x[:n] # skip slack bus (on polar, f is not used)
            self.v[1:] += delta_x[n:] # skip slack bus (on polar, e is not used)
            
            print(f"Iteration {iteration + 1}: Max mismatch = {max_mismatch:.6e}")

        print(f"❌ Did not converge. Max mismatch: {max_mismatch:.6e}")
        return False
        
    # def _update_results(self):
    #     # final p_calc, q_calc update
    #     p_final, q_final = self._calculate_current_power()
        
    #     self.df['e'] = self.e
    #     self.df['f'] = self.f
    #     self.df['V_mag'] = np.sqrt(self.e**2 + self.f**2)
    #     self.df['V_angle'] = np.degrees(np.arctan2(self.f, self.e))
    #     self.df['P_calc'] = p_final
    #     self.df['Q_calc'] = q_final
        
    #     print(self.df)
    
    # def get_final_results(self, ENERGY_BALANCE_TOLERANCE) :
    #     # Update p_calc and q_calc based on the final converged e and f values.
    #     self._calculate_current_power()
        
    #     V_mag = np.sqrt(self.e**2 + self.f**2)
    #     V_angle = np.degrees(np.arctan2(self.f, self.e))
        
    #     # mag and angle results per line
    #     bus_results_data = {
    #         'Bus_ID': self.df['ID'].values,
    #         'V_mag': V_mag,
    #         'V_angle_degree': V_angle,
    #         'P_calc': self.p_calc,
    #         'Q_calc': self.q_calc
    #     }
    #     bus_results_df = pd.DataFrame(bus_results_data)
        
    #     # calculated to line loss
    #     line_flow_df, total_line_loss_p = self._calculate_line_flows()
        
    #     # value1 - total Load
    #     total_load_p = self.df['P_L'].sum()
    #     total_load_q = self.df['Q_L'].sum()
        
    #     # value2 - slack bus injection amount (Bus-type 1)
    #     slack_idx = self.df[self.df['Bus-type'] == 1].index
    #     slack_p = self.p_calc[slack_idx].sum()
    #     slack_q = self.q_calc[slack_idx].sum()
        
    #     # value3 - determining whether energy is conserved (Gen = Load + Loss)
    #     total_gen_p = self.p_calc.sum() + self.df['P_L'].sum()
    #     is_p_balanced = np.isclose( slack_p + self.df['P_g'].sum(), 
    #                                 total_load_p + total_line_loss_p, atol = ENERGY_BALANCE_TOLERANCE )
        
    #     # result dataframe
    #     summary_data = {
    #         'Total_Line_Loss_P' : [total_line_loss_p],
    #         'Total_Load_P' : [total_load_p],
    #         'Slack_Injection_P' : [slack_p],
    #         'Energy_Balance_Check' : [is_p_balanced]
    #     }
        
    #     return pd.DataFrame(summary_data), bus_results_df, line_flow_df
        
    # def _calculate_line_flows(self) :
    #     # initial setting
    #     line_results = []
    #     total_loss_p = 0
        
    #     for _, row in self.line_df.iterrows():
    #         # extract to index (1-based to 0-based)
    #         from_idx = int(row['ID_from']) - 1
    #         to_idx = int(row['ID_to']) - 1

    #         # Bus voltage at both ends (complex type)
    #         Vi = complex(self.e[from_idx], self.f[from_idx])
    #         Vj = complex(self.e[to_idx], self.f[to_idx])

    #         # line admittance  y_ij = 1 / (R + jX)
    #         z_line = complex(row['R'], row['X'])
    #         y_ij = 1 / z_line
    #         b_shunt = complex(0, row['B'] / 2) # charging components

    #         # From -> To flowing current : I_ij = (Vi - Vj)*y_ij + Vi*b_shunt
    #         I_ij = (Vi - Vj) * y_ij + Vi * b_shunt
    #         S_ij = Vi * np.conj(I_ij) # transmission power

    #         # To -> From flowing current : I_ji = (Vj - Vi)*y_ij + Vj*b_shunt
    #         I_ji = (Vj - Vi) * y_ij + Vj * b_shunt
    #         S_ji = Vj * np.conj(I_ji) # reception power

    #         # Line loss = (transmission P + reception P) (부호가 반대이므로 합치면 손실만 남음)
    #         loss_p = S_ij.real + S_ji.real
    #         total_loss_p += loss_p

    #         line_results.append({
    #             'from': from_idx + 1,
    #             'p_send': S_ij.real,
    #             'to': to_idx + 1,
    #             'p_recv': -S_ji.real, # From a reception power perspective, sign reversal
    #             'loss': loss_p
    #         })

    #     return pd.DataFrame(line_results), total_loss_p