from modules.data_loader import DataLoader
from modules.pf_solver import PowerFlowLoader

from config import BUS_FILE_PATH, LINE_FILE_PATH, BUS_FILE_HEADER_LIST, LINE_FILE_HEADER_LIST

class FlowManager : 
    def run_power_flow(self) : 
        # data load
        loader = DataLoader(BUS_FILE_PATH, LINE_FILE_PATH, BUS_FILE_HEADER_LIST, LINE_FILE_HEADER_LIST)
        loader.load_all_data()
        # make the y_bus matrix
        y_bus_matrix = loader.make_admittance_matrix()
        
        print(y_bus_matrix)

if __name__ == "__main__":
    manager = FlowManager()
    manager.run_power_flow()