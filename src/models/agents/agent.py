from __future__ import annotations
import uuid
import threading
import ctypes

from ...utils.custom_objects.data_classes import Solution


class ParticleAgent(threading.Thread):
    def __init__(self, supervisor: 'Supervisor'):
        self.agent_id = uuid.uuid4()
        super().__init__(daemon=True)
        self.event = threading.Event()
        self.supervisor: 'Supervisor' = supervisor
        self.local_best: Solution = Solution()
        self.global_best_agent_type: Solution = Solution()
    
    def __hash__(self):
        return hash(self.agent_id)

    def __eq__(self, other):
        return self.local_best.value == other.local_best.value
    
    def __lt__(self, other):
        return self.local_best.value < other.local_best.value
    
    def __gt__(self, other):
        return self.local_best.value > other.local_best.value
    
    def execute(self) -> None:
        pass
    
    def set_global_best(self, global_best: Solution):
        pass
    
    def set_global_best_agent_type(self, global_best: Solution):
        self.global_best_agent_type = global_best

    def adapt(self, exploration: int, exploitation: int):
        pass

    def run(self):
        while True:
            self.event.wait()
            self.execute()
            self.event.clear()
            self.supervisor.agent_stopped(self.__class__)
    
    def go(self):
        self.event.set()
        
    def stop(self):
        self.event.clear()
    
    def kill(self):
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(self.ident), ctypes.py_object(SystemExit)
        )