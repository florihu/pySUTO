class Process:
    def __init__(self, process_name: str, process_id: int):
        self.process_name = process_name
        self.process_id = process_id

    def __repr__(self):
        return f"PROCESS(process_name={self.process_name}, process_id={self.process_id})"