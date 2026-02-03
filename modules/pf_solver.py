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
        
    def get_final_results(self, ENERGY_BALANCE_TOLERANCE) : 
        # final voltage magnitude and angle in degrees
        V_final = self.v
        delta_deg = np.degrees(self.delta)
        
        # line loss calculation (Line by line)
        total_p_loss = 0.0
        total_q_loss = 0.0
        V_complex = self.v * np.exp(1j * self.delta) # V = |V|*e^(jδ) = |V|*(cosδ + jsinδ)
        
        for _, line in self.line_df.iterrows() :
            # index adjustment (ID starts from 1)
            i, j = int(line['ID_from']) - 1, int(line['ID_to']) - 1
            # line parameters (R, X, B)
            Z = line['R'] + 1j * line['X']
            B_half = 1j * (line['B'] / 2)
            
            # step1 : calculate line current from bus i to j and j to i --> Iij = (Vi - Vj)/Z + B/2*Vi
            Iij = (V_complex[i] - V_complex[j]) / Z + B_half * V_complex[i]
            Iji = (V_complex[j] - V_complex[i]) / Z + B_half * V_complex[j]
            
            # step2 : calculate complex power flow Sij and Sji --> Sij = Vi * conj(Iij), Sji = Vj * conj(Iji)
            Sij = V_complex[i] * np.conj(Iij)
            Sji = V_complex[j] * np.conj(Iji)
            
            line_loss = Sij + Sji
            total_p_loss += line_loss.real
            total_q_loss += line_loss.imag
    
        # total load calculation
        total_p_load = np.sum(self.df['P_L'].values)
        total_q_load = np.sum(self.df['Q_L'].values)
        
        # slack bus injection calculation
        slack_p_inj = self.p_calc[0] + self.df.loc[0, 'P_L']
        slack_q_inj = self.q_calc[0] + self.df.loc[0, 'Q_L']
        
        # energy balance check --> Injection = Load + Loss
        total_p_gen = np.sum(self.p_calc + self.df['P_L'].values)
        total_q_gen = np.sum(self.q_calc + self.df['Q_L'].values)
        
        energy_p_check = abs(total_p_gen - (total_p_load + total_p_loss)) < ENERGY_BALANCE_TOLERANCE
        energy_q_check = abs(total_q_gen - (total_q_load + total_q_loss)) < ENERGY_BALANCE_TOLERANCE
        is_conserved = energy_p_check and energy_q_check
        
        # make results DataFrame
        bus_results = pd.DataFrame({
            'ID' : self.df['ID'],
            'BUS' : range(1, len(self.df) + 1),
            '|V| (p.u.)' : V_final,
            'δ (deg)' : delta_deg,
            'P_calc (p.u.)' : self.p_calc,
            'Q_calc (p.u.)' : self.q_calc
        })
        
        summary_results = {
            'Total P Load (p.u.)' : total_p_load,
            'Total Q Load (p.u.)' : total_q_load,
            'Total P Loss (p.u.)' : total_p_loss,
            'Total Q Loss (p.u.)' : total_q_loss,
            'Slack P Injection (p.u.)' : slack_p_inj,
            'Slack Q Injection (p.u.)' : slack_q_inj,
            'Energy Conserved' : is_conserved
        }
        
        # print("\n### Bus Results ###")
        # print(bus_results)
        # print("\n### Summary Results ###")
        # for key, value in summary_results.items():
        #     print(f"{key}: {value}")
            
        return bus_results, summary_results