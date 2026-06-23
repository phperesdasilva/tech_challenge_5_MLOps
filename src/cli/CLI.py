from eda.DataManager import DataManager
from event_generator.SyntheticEventGenerator import SyntheticEventGenerator

class CLI:
    def __init__(self):
        self.data_manager = DataManager()
        self.event_generator = SyntheticEventGenerator()

    def run_eda(self):
        self.data_manager.run_eda()

    def generate_events(self):
        self.event_generator.generate_events()
