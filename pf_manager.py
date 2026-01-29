from modules.data_loader import DataLoader

from config import BUS_FILE_PATH, LINE_FILE_PATH, BUS_FILE_HEADER_LIST, LINE_FILE_HEADER_LIST

class FlowManager : 
    def run_power_flow(self) : 
        loader = DataLoader(BUS_FILE_PATH, LINE_FILE_PATH, BUS_FILE_HEADER_LIST, LINE_FILE_HEADER_LIST)
        
        busDataFrame = loader.read_data_file(BUS_FILE_PATH, BUS_FILE_HEADER_LIST)
        lineDataFrame = loader.read_data_file(LINE_FILE_PATH, LINE_FILE_HEADER_LIST)
        
        print(busDataFrame)

if __name__ == "__main__":
    manager = FlowManager()
    manager.run_power_flow()