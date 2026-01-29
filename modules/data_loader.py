import pandas as pd
    
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
    def read_data_file(self, path, headers) :
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
        self.bus_df = self._load_and_clean(self.busFilePath, self.busHeader)
        self.line_df = self._load_and_clean(self.lineFilePath, self.lineHeader)
        
        return self.bus_df, self.line_df
        
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
        
    # def make_admittance_matrix() :
        