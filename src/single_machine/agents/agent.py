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
        self.local_best: Solution | None = None
    
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
        self.global_best = global_best

    def run(self):
        while True:
            self.event.wait()
            self.execute()
            self.event.clear()
    
    def go(self):
        self.event.set()
        
    def stop(self):
        self.event.clear()
    
    def kill(self):
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(self.ident), ctypes.py_object(SystemExit)
        )