import pandas as pd
    
class DataLoader :
    def __init__(self, busFilePath, lineFilePath, busHeader, lineHeader) :
        self.busFilePath = busFilePath
        self.lineFilePath = lineFilePath
        self.busHeader = busHeader
        self.lineHeader = lineHeader
    
    def read_data_file(self, file_path, headers) :
        try : 
            df = pd.read_csv(file_path, sep='\t', header=None, dtype='a')
            
            if len(headers) == len(df.columns) : # add header list to dataframe
                df.columns = headers
                return df
            else : 
                print("please check the data files and Header information.")
                return None
            
        except Exception as e :
            print(f"An error occurred while reading the file : {e}")
            return None
            
    # def make_dataframe(self, bud_header) : 
    
    # def distinguish_input_unknown_data() : 
        
    # def make_admittance_matrix() :
        
