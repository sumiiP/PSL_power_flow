import pandas as pd
import numpy as np

class PowerFlowLoader : 
    def __init__(self, y_bus, bus_df, id_info) : 
        self.Y = y_bus
        self.df = bus_df
        
        # mismatch judgment index
        self.p_idx = [int(i) - 1 for i in id_info['p_id']]
        self.q_idx = [int(i) - 1 for i in id_info['q_id']]
        
        # initial setting
        self.e = bus_df['e'].values.astype(float)
        self.f = bus_df['f'].values.astype(float)
        self.p_net = bus_df['P_net'].values
        self.q_net = bus_df['Q_net'].values

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
        
        p_calc = e * term1 + f * term2
        q_calc = f * term1 - e * term2
        
        return p_calc, q_calc, term1, term2
        
    def _make_mismatch_vector(self) : # calculate to delta y
        p_calc, q_calc = self._calculate_current_power()
        
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
                break
            
            # step 3
            J = self._make_jacobian()
            
            # step 4 : (J * dx = mismatch)
            dx = np.linalg.solve(J, mismatch)
            
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
                if i == j : # Diagonal
                    J11_f[i, i] = term1[i] + G[i, i]*e[i] + B[i, i]*f[i]
                    J12_f[i, i] = term2[i] + G[i, i]*f[i] - B[i, i]*e[i]
                    J21_f[i, i] = term2[i] - (G[i, i]*f[i] - B[i, i]*e[i])
                    J22_f[i, i] = -(term1[i] - (G[i, i]*e[i] + B[i, i]*f[i]))
                    
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