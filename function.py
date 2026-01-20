import numpy as np
import pandas as pd

def read_data_file(file_path, headers) :
    try : 
        RESULT_MATRIX = pd.read_csv(file_path, sep='\t', header=None)

        # row,columns shape 확인
        rows, cols = RESULT_MATRIX.shape
        
        # header 추가
        if len(headers) == int(cols) :
            RESULT_MATRIX.columns = headers
            return RESULT_MATRIX
        else : 
            print("Header information error.")
            return None
        
    except Exception as e : 
        print(f"An error occurred while reading the file : {e}")
        return None    

def input_matrix() :
    while True : 
        try : 
            # Received data for matrix A from the user
            MAT_VALUE = input("Enter matrix in a single line the left (ex. 1,2,3,...) :")
            MAT_ROW = input("Enter the 'column' form of matrix A (ex. 5): ")
            
            # datatype preprocessing (string to int)
            number_list_A = [float(x.strip()) for x in MAT_VALUE.split(',')]
                # [TBD] edit by deleting spaces!!!
            
            # matrix form reshape
            MATRIX = np.array(number_list_A).reshape(-1, int(MAT_ROW))
            
            #error except process : TBD
            
            return MATRIX
        
        except ValueError :
            print("[ERROR] : Please enter only numbers and follow the format. try agiain")
            
        except Exception as e : 
            print(f"[ERROR]_{e} : Please enter only numbers and follow the format. try agiain")
            
def matrix_judgment (mat_a, mat_b) :
    if mat_a.shape[1] == mat_b.shape[0]:
        return True
    else : 
        print(f"Not possible to multipulication: A_column({mat_a.shape[1]}) != B_row({mat_b.shape[0]})")
        return False

def matrix_multipulication(mat_a, mat_b) :
    RESULT = np.matmul(mat_a, mat_b)
    
    return RESULT