import pandas as pd
    
class DataLoader :
    def __init__(self, busFilePath, lineFilePath, busHeader, lineHeader) :
        self.busFilePath = busFilePath
        self.lineFilePath = lineFilePath
        self.busHeader = busHeader
        self.lineHeader = lineHeader
    # %%
    def read_data_file(self, file_path, headers) :
        try : 
            df = pd.read_csv(file_path, sep='\t', header=None)
            df = df.apply(pd.to_numeric, downcast='float', errors = 'coerce')
            
            if len(headers) == len(df.columns) : # add header list to dataframe
                df.columns = headers
                return df
            
            else : 
                print("please check the data files and Header information.")
                return None
            
        except Exception as e :
            print(f"An error occurred while reading the file : {e}")
            return None
    
    def distinguish_input_unknown_data(self, busDataFrame) : 
        
        busDataFrame['P_net'] = busDataFrame['P_g'] - busDataFrame['P_L'] #calculate Net active Power
        busDataFrame['Q_net'] = busDataFrame['Q_g'] - busDataFrame['Q_L'] #calculate Net reactive Power
        
        non_slack_id_list = busDataFrame[busDataFrame['Bus-type'] != 1]['ID'].tolist() #P mismatch judgment index
        pd_bus_id_list = busDataFrame[busDataFrame['Bus-type'] == 3]['ID'].tolist() #Q mismatch judgment index
        
        id_info = {
            'p_id' : non_slack_id_list,
            'q_id' : pd_bus_id_list
        }
        
        return busDataFrame, id_info
        
    def make_admittance_matrix() :
        