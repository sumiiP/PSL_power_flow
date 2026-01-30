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
        
        return p_calc, q_calc
        
    def _make_mismatch_vector(self) : # calculate to delta y
        
        p_calc, q_calc = self._calculate_current_power()
        
        # active power(P) mismatch
        delta_p = self.p_net[self.p_idx] - p_calc[self.p_idx]
        # reactive power(Q) mismatch
        delta_q = self.q_net[self.q_idx] - q_calc[self.q_idx]
        
        # The upper part is the differentiation with respect to P, and the lower part is the differentiation with respect to Q.
        mismatch_vector = np.concatenate([delta_p, delta_q])
        
        return mismatch_vector
    
    def calculate_power() :
        
    
    
# $P_{calc}, Q_{calc}$를 구하는 복소수 벡터 연산 로직 (Mismatch 벡터의 재료)
# 자코비안 행렬의 각 요소($H, N, M, L$)를 채우는 미분 공식 구현
# 전체 Newton-Raphson 루프의 정지 조건(Tolerance) 설정