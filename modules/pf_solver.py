import pandas as pd
import numpy as np

class PowerFlowLoader : 
    def __init__(self, y_bus, bus_df, id_info, line_df) : 
        self.Y = y_bus
        self.df = bus_df
        
        # mismatch judgment index
        self.p_idx = [int(i) - 1 for i in id_info['p_id']]
        self.q_idx = [int(i) - 1 for i in id_info['q_id']]
        
        # initial setting
        self.e = bus_df['e'].values.astype(float)
        self.f = bus_df['f'].values.astype(float)
        
        # base_mva = 100.0
        self.p_net = bus_df['P_net'].values
        self.q_net = bus_df['Q_net'].values
        self.p_calc = None
        self.q_calc = None
        
        # Used to calculate final results
        self.line_df = line_df
        
    def _calculate_current_power(self) :
        # Separation of real part(G) and imaginary part(B) of Y_bus
        G = self.Y.real
        B = self.Y.imag
        e = self.e
        f = self.f
        
        # Power calculataion in ractangular coordinates
        # I_real = Ge - Bf, I_imag = Gf + Be
        term1 = G @ e - B @ f
        term2 = G @ f + B @ e
        
        self.p_calc = e * term1 + f * term2
        self.q_calc = f * term1 - e * term2
        
        return self.p_calc, self.q_calc, term1, term2
        
    def _make_mismatch_vector(self) : # calculate to delta y
        p_calc, q_calc, _, _ = self._calculate_current_power()
        
        # active power(P) mismatch
        delta_p = self.p_net[self.p_idx] - p_calc[self.p_idx]
        # reactive power(Q) mismatch
        delta_q = self.q_net[self.q_idx] - q_calc[self.q_idx]
        
        # The upper part is the differentiation with respect to P, and the lower part is the differentiation with respect to Q.
        mismatch_vector = np.concatenate([delta_p, delta_q])
        
        return mismatch_vector
    
    def calculate_power(self, MAX_ITER, TOLERANCE) :
        # Newton-Raphson Iteration loop
        for i in range(MAX_ITER) : 
            mismatch = self._make_mismatch_vector()
            
            # step 1
            max_error = np.max(np.abs(mismatch))
            print(f"Iteration {i+1}: Max Mismatch = {max_error:.8f}")

            # step 2
            if max_error < TOLERANCE : 
                print("--- Power Flow Converged! ---")
                return True
            
            # step 3
            J = self._make_jacobian()
            
            # step 4 : (J * dx = mismatch)
            dx = np.linalg.solve(J, mismatch)
            # print(f"dx {i}번째 : {dx}")
            
            # step 5
            len_p = len(self.p_idx) # delta P num
            len_q = len(self.q_idx) # delta Q num
            
            # step 6
            df_update = dx[:len_p] # delta f
            de_update = dx[len_p:] # delta e
            
            # step 7
            self.f[self.p_idx] += df_update
            self.e[self.q_idx] += de_update
            
        else : 
            print("!!! Power Flow Did Not Coverage !!!")
            return False
    
    def _make_jacobian(self) : 
        # initial setting
        e = self.e
        f = self.f
        G = self.Y.real
        B = self.Y.imag
        
        # term1 = I_real = Ge - Bf, term2 = I_imag = Gf + Be
        _, _, term1, term2 = self._calculate_current_power()
        num_bus = len(self.df)
        
        # J11 = dP/df, J12 = dP/de, J21 = dQ/df, J22 = dQ/de
        J11_f = np.zeros((num_bus, num_bus))
        J12_f = np.zeros((num_bus, num_bus))
        J21_f = np.zeros((num_bus, num_bus))
        J22_f = np.zeros((num_bus, num_bus))
        
        # i != j (Off-diagonal), i = j (Diagonal)
        for i in range(num_bus) : 
            for j in range(num_bus) :
                if i == j: # Diagonal
                    # dPi/dfi
                    J11_f[i, i] = term1[i] + G[i, i]*f[i] - B[i, i]*e[i]
                    # dPi/dei
                    J12_f[i, i] = term2[i] + G[i, i]*e[i] + B[i, i]*f[i]
                    # dQi/dfi
                    J21_f[i, i] = term2[i] - (G[i, i]*e[i] + B[i, i]*f[i])
                    # dQi/dei
                    J22_f[i, i] = -term1[i] + (G[i, i]*f[i] - B[i, i]*e[i])
                # if i == j : # Diagonal
                #     J11_f[i, i] = term1[i] + G[i, i]*e[i] - B[i, i]*f[i]
                #     J12_f[i, i] = term2[i] + G[i, i]*f[i] + B[i, i]*e[i]
                #     J21_f[i, i] = term2[i] - (G[i, i]*f[i] + B[i, i]*e[i])
                #     J22_f[i, i] = -term1[i] + (G[i, i]*e[i] - B[i, i]*f[i])
                    
                else : # Off-diagonal
                    J11_f[i, j] = G[i, j]*e[i] + B[i, j]*f[i]
                    J12_f[i, j] = G[i, j]*f[i] - B[i, j]*e[i]
                    J21_f[i, j] = J12_f[i, j]
                    J22_f[i, j] = -J11_f[i, j]
        
        # extract the sub matrix by p_dix, q_idx
        J11 = J11_f[np.ix_(self.p_idx, self.p_idx)]
        J12 = J12_f[np.ix_(self.p_idx, self.q_idx)]
        J21 = J21_f[np.ix_(self.q_idx, self.p_idx)]
        J22 = J22_f[np.ix_(self.q_idx, self.q_idx)]
        
        # final Jacobian matrix            
        J = np.block([
            [J11, J12],
            [J21, J22]
        ])
        
        return J
    
    def get_final_results(self, ENERGY_BALANCE_TOLERANCE) :
        # Update p_calc and q_calc based on the final converged e and f values.
        self._calculate_current_power()
        
        V_mag = np.sqrt(self.e**2 + self.f**2)
        V_angle = np.degrees(np.arctan2(self.f, self.e))
        
        # mag and angle results per line
        bus_results_data = {
            'Bus_ID': self.df['ID'].values,
            'V_mag': V_mag,
            'V_angle_degree': V_angle,
            'P_calc': self.p_calc,
            'Q_calc': self.q_calc
        }
        bus_results_df = pd.DataFrame(bus_results_data)
        
        # calculated to line loss
        line_flow_df, total_line_loss_p = self._calculate_line_flows()
        
        # value1 - total Load
        total_load_p = self.df['P_L'].sum()
        total_load_q = self.df['Q_L'].sum()
        
        # value2 - slack bus injection amount (Bus-type 1)
        slack_idx = self.df[self.df['Bus-type'] == 1].index
        slack_p = self.p_calc[slack_idx].sum()
        slack_q = self.q_calc[slack_idx].sum()
        
        # value3 - determining whether energy is conserved (Gen = Load + Loss)
        total_gen_p = self.p_calc.sum() + self.df['P_L'].sum()
        is_p_balanced = np.isclose( slack_p + self.df['P_g'].sum(), 
                                    total_load_p + total_line_loss_p, atol = ENERGY_BALANCE_TOLERANCE )
        
        # result dataframe
        summary_data = {
            'Total_Line_Loss_P' : [total_line_loss_p],
            'Total_Load_P' : [total_load_p],
            'Slack_Injection_P' : [slack_p],
            'Energy_Balance_Check' : [is_p_balanced]
        }
        
        return pd.DataFrame(summary_data), bus_results_df, line_flow_df
        
    def _calculate_line_flows(self) :
        # initial setting
        line_results = []
        total_loss_p = 0
        
        for _, row in self.line_df.iterrows():
            # extract to index (1-based to 0-based)
            from_idx = int(row['ID_from']) - 1
            to_idx = int(row['ID_to']) - 1

            # Bus voltage at both ends (complex type)
            Vi = complex(self.e[from_idx], self.f[from_idx])
            Vj = complex(self.e[to_idx], self.f[to_idx])

            # line admittance  y_ij = 1 / (R + jX)
            z_line = complex(row['R'], row['X'])
            y_ij = 1 / z_line
            b_shunt = complex(0, row['B'] / 2) # charging components

            # From -> To flowing current : I_ij = (Vi - Vj)*y_ij + Vi*b_shunt
            I_ij = (Vi - Vj) * y_ij + Vi * b_shunt
            S_ij = Vi * np.conj(I_ij) # transmission power

            # To -> From flowing current : I_ji = (Vj - Vi)*y_ij + Vj*b_shunt
            I_ji = (Vj - Vi) * y_ij + Vj * b_shunt
            S_ji = Vj * np.conj(I_ji) # reception power

            # Line loss = (transmission P + reception P) (부호가 반대이므로 합치면 손실만 남음)
            loss_p = S_ij.real + S_ji.real
            total_loss_p += loss_p

            line_results.append({
                'from': from_idx + 1,
                'p_send': S_ij.real,
                'to': to_idx + 1,
                'p_recv': -S_ji.real, # From a reception power perspective, sign reversal
                'loss': loss_p
            })

        return pd.DataFrame(line_results), total_loss_p