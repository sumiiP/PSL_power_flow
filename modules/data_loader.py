import pandas as pd
import numpy as np
    
class DataLoader :
    def __init__(self, busFilePath, lineFilePath, busHeader, lineHeader) :
        self.busFilePath = busFilePath
        self.lineFilePath = lineFilePath
        self.busHeader = busHeader
        self.lineHeader = lineHeader
        
        #initalize the dataframes
        self.bus_df = None
        self.line_df = None
        
    # %%
    def _read_data_file(self, path, headers) :
        try : 
            df = pd.read_csv(path, sep='\t', header=None)
            df = df.apply(pd.to_numeric, downcast='float', errors = 'coerce')
            
            if len(headers) == len(df.columns) : # add header list to dataframe
                df.columns = headers
                return df
            
            else : 
                print(f"Header mismatch in {path}")
                return None
            
        except Exception as e :
            print(f"Error reading {path}: {e}")
            return None
        
    def load_all_data(self) :
        self.bus_df = self._read_data_file(self.busFilePath, self.busHeader)
        self.line_df = self._read_data_file(self.lineFilePath, self.lineHeader)
        
    def distinguish_input_unknown_data(self) : 
        df = self.bus_df
        df['P_net'] = df['P_g'] - df['P_L'] #calculate Net active Power
        df['Q_net'] = df['Q_g'] - df['Q_L'] #calculate Net reactive Power
        
        non_slack_id_list = df[df['Bus-type'] != 1]['ID'].tolist() #P mismatch judgment index
        pd_bus_id_list = df[df['Bus-type'] == 3]['ID'].tolist() #Q mismatch judgment index
        
        id_info = {
            'p_id' : non_slack_id_list,
            'q_id' : pd_bus_id_list
        }
        
        return id_info
    
    def convert_to_rectangular(self) :
        # Convert polar coordinates (V, delta) to rectangular coordinates (e, f)
        # V = |V|*cos(δ) + j*|V|*sin(δ)
        self.bus_df['e'] = self.bus_df['|Vi|'] * np.cos(np.radians(self.bus_df['δi']))
        self.bus_df['f'] = self.bus_df['|Vi|'] * np.sin(np.radians(self.bus_df['δi']))
        
    def make_admittance_matrix(self) : # [TBD] Improvments via Numpy's vectorized operations or sparse matrix library
        if self.bus_df is None or self.line_df is None : 
            self.load_all_data()
        
        bus_df = self.bus_df
        line_df = self.line_df
        
        #initalize the Y_bus matrix to zeros
        bus_num = len(bus_df)
        Y_bus = np.zeros((bus_num, bus_num), dtype=complex)
        
        for _, row in line_df.iterrows() :
            k = int(row['ID_from']) - 1
            n = int(row['ID_to']) - 1
            
            # Z = R + jX, Y = 1/Z
            z_line = row['R'] + 1j * row['X']
            # z_line = complex(row['R'], row['X'])
            y_line = 1 / z_line
            
            # shunt component (B/2 placed on each end bus)
            b_shunt = complex(0, row['B'] / 2)
            
            # Mutual Admittance : Y_kn = Y_nk = -y_line
            Y_bus[k, n] -= y_line
            Y_bus[n, k] -= y_line
            
            # Self Admittance : Cumulative sum of connected y_lines and shunts.
            # Y_kk = sum(y_connected) + sum(y_shunt)
            Y_bus[k, k] += (y_line + b_shunt)
            Y_bus[n, n] += (y_line + b_shunt)
        
        # # numpy matrix to DataFrame
        # bus_ids = bus_df['ID'].astype(int).tolist()
        # y_bus_df = pd.DataFrame(Y_bus, index=bus_ids, columns=bus_ids)
        
        # # print setting
        # pd.options.display.float_format = '{:.4f}'.format
        # print("\n### Y_bus Matrix (Admittance) ###")
        # print(y_bus_df)
        
        return Y_bus