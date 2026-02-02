from modules.data_loader import DataLoader
from modules.pf_solver import PowerFlowLoader

from config import BUS_FILE_PATH, LINE_FILE_PATH, BUS_FILE_HEADER_LIST, LINE_FILE_HEADER_LIST
from config import MAX_ITER, TOLERANCE, ENERGY_BALANCE_TOLERANCE

class FlowManager : 
    def run_power_flow(self) : 
        # data load
        loader = DataLoader(BUS_FILE_PATH, LINE_FILE_PATH, BUS_FILE_HEADER_LIST, LINE_FILE_HEADER_LIST)
        loader.load_all_data()
        
        id_info = loader.distinguish_input_unknown_data()
        loader.convert_to_rectangular() # convert to rectangular coodinates
        
        y_bus = loader.make_admittance_matrix()

        # calculate the power flow equation
        # solver = PowerFlowLoader(y_bus, loader.bus_df, id_info, loader.line_df)
        # solver.calculate_power(MAX_ITER, TOLERANCE)
        
        # # # data processing
        # summary_df, bus_df, line_df = solver.get_final_results(ENERGY_BALANCE_TOLERANCE)
        
        # print(f"total df : {summary_df}")
        # print(f"bus df : {bus_df}")
        # print(f"line df : {line_df}")
        # export to csv and graph
        

if __name__ == "__main__":
    manager = FlowManager()
    manager.run_power_flow()