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
        # num_bus = len(bus_df)
        # self.e = np.ones(num_bus)
        # self.f = np.zeros(num_bus)
        self.e = bus_df['e'].values.astype(float)
        self.f = bus_df['f'].values.astype(float)
        
        self.p_net = bus_df['P_net'].values
        self.q_net = bus_df['Q_net'].values
        self.p_calc = None
        self.q_calc = None
        
        # Used to calculate final results
        self.line_df = line_df
        
    def _calculate_current_power(self) :
        # Calculate the current power (P_calc, Q_calc) based on the current e and f values.
        V = self.e + 1j * self.f  # Complex voltage vector
        I = self.Y @ V            # Complex current vector
        S = V * np.conj(I)        # Complex power vector
        
        self.p_calc = S.real
        self.q_calc = S.imag
        
        return self.p_calc, self.q_calc
    
    def _make_mismatch_vector(self) : # calculate to delta y
        self._calculate_current_power()
        
        # Mismatch vector calculation
        delta_p = self.p_net[self.p_idx] - self.p_calc[self.p_idx]
        delta_q = self.q_net[self.q_idx] - self.q_calc[self.q_idx]
        
        mismatch_vector = np.concatenate([delta_p, delta_q])
        
        return mismatch_vector
    
    def _make_jacobian(self):
        # 1. Initial settings
        e, f = self.e, self.f
        V = e + 1j * f
        G, B = self.Y.real, self.Y.imag

        # 2. Calculate current I
        I = self.Y @ V
        Ir, Ii = I.real, I.imag

        # 3. Construct full Jacobian matrices
        # J11=dP/df, J12=dP/de, J21=dQ/df, J22=dQ/de
        diag_e = np.diag(e) 
        diag_f = np.diag(f)
        diag_Ir = np.diag(Ir)
        diag_Ii = np.diag(Ii)

        # dP/df (J11), dP/de (J12)
        J11_f = diag_e @ B - diag_f @ G + diag_Ii  # dP/df
        J12_f = diag_e @ G + diag_f @ B + diag_Ir  # dP/de

        # dQ/df (J21), dQ/de (J22)
        J21_f = diag_f @ B + diag_e @ G - diag_Ir  # dQ/df
        J22_f = diag_f @ G - diag_e @ B + diag_Ii  # dQ/de

        # 4. Extract sub-matrices based on p_idx and q_idx
        J11 = J11_f[np.ix_(self.p_idx, self.p_idx)]
        J12 = J12_f[np.ix_(self.p_idx, self.q_idx)]
        J21 = J21_f[np.ix_(self.q_idx, self.p_idx)]
        J22 = J22_f[np.ix_(self.q_idx, self.q_idx)]

        J = np.block([[J11, J12], [J21, J22]])
        print("Jacobian Matrix:\n", J)
        
        return np.block([[J11, J12], [J21, J22]])
    
    # def _make_jacobian(self):
    #     # initial setting
    #     e = self.e
    #     f = self.f
    #     G = self.Y.real
    #     B = self.Y.imag
    #     # num_bus = len(self.df)
    #     n_p = len(self.p_idx)
    #     n_q = len(self.q_idx)

    #     # I = (G + jB)(e + jf) = (Ge - Bf) + j(Gf + Be)
    #     I_real = G @ e - B @ f
    #     I_imag = G @ f + B @ e

    #     # Jacobian sub matrix zero initalize
    #     J11 = np.zeros((n_p, n_p)) # dP/df
    #     J12 = np.zeros((n_p, n_q)) # dP/de
    #     J21 = np.zeros((n_q, n_p)) # dQ/df
    #     J22 = np.zeros((n_q, n_q)) # dQ/de

        # for i in range(len(self.df)):
        #     for j in range(len(self.df)):
        #         if i == j:  # Diagonal 
        #             J11_f[i, i] = f[i] * G[i, i] - e[i] * B[i, i] + I_imag[i]
        #             J12_f[i, i] = e[i] * G[i, i] + f[i] * B[i, i] + I_real[i]
        #             J21_f[i, i] = f[i] * B[i, i] + e[i] * G[i, i] - I_real[i]
        #             J22_f[i, i] = -e[i] * B[i, i] + f[i] * G[i, i] + I_imag[i]
                    
        #         else:  # Off-diagonal 
        #             J11_f[i, j] = f[i] * G[i, j] - e[i] * B[i, j]
        #             J12_f[i, j] = e[i] * G[i, j] + f[i] * B[i, j]
        #             J21_f[i, j] = f[i] * B[i, j] + e[i] * G[i, j]
        #             J22_f[i, j] = -e[i] * B[i, j] + f[i] * G[i, j]

        # # extract the sub matrix by p_dix, q_idx
        # J11 = J11_f[np.ix_(self.p_idx, self.p_idx)]
        # J12 = J12_f[np.ix_(self.p_idx, self.q_idx)]
        # J21 = J21_f[np.ix_(self.q_idx, self.p_idx)]
        # J22 = J22_f[np.ix_(self.q_idx, self.q_idx)]

        # final Jacobian matrix 
    
    def calculate_power(self, MAX_ITER, TOLERANCE) :
        # Newton-Raphson Iteration loop
        for iteration in range(MAX_ITER) :
            # 1. Mismatch vector calculation
            mismatch_vector = self._make_mismatch_vector()
            
            # 2. Convergence check
            max_mismatch = np.max(np.abs(mismatch_vector))
            if max_mismatch < TOLERANCE :
                print(f"Converged in {iteration} iterations. Max mismatch: {max_mismatch}")
                return True
            
            # 3. Jacobian matrix calculation
            J = self._make_jacobian()
            
            # 4. Solve for state variable updates
            delta_x = np.linalg.solve(J, mismatch_vector)
            
            # # delta_x 계산 직후에 추가
            # max_step = np.max(np.abs(delta_x))
            # print(f"- Max Voltage Update Step: {max_step:.4f}")
            
            # if max_step > 0.5:
            #     print("  ⚠️ Warning: Update step is too large! Check units (MW vs p.u.) or Jacobian signs.")
                
            # 5. Update state variables (e, f)
            # Update e and f based on the indices
            num_p = len(self.p_idx)
            delta_f = delta_x[:num_p]
            delta_e = delta_x[num_p:]

            self.f[self.p_idx] += delta_f
            self.e[self.q_idx] += delta_e
            
        print(f"Did not converge within {MAX_ITER} iterations. Max mismatch: {max_mismatch}")
        return False
        
    def _update_results(self):
        # final p_calc, q_calc update
        p_final, q_final = self._calculate_current_power()
        
        self.df['e'] = self.e
        self.df['f'] = self.f
        self.df['V_mag'] = np.sqrt(self.e**2 + self.f**2)
        self.df['V_angle'] = np.degrees(np.arctan2(self.f, self.e))
        self.df['P_calc'] = p_final
        self.df['Q_calc'] = q_final
        
        print(self.df)
    
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