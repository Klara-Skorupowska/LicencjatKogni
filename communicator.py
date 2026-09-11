import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

class Communicator:
    """Simulates ROS Topics and Services."""
    def __init__(self, max_workers=10):
        self.topics = {}
        self.services = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    # --- Topics (Publish/Subscribe) ---
    def subscribe(self, topic, callback):
        with self.lock:
            if topic not in self.topics:
                self.topics[topic] = []
            self.topics[topic].append(callback)

    def publish(self, topic, data):
        with self.lock:
            callbacks = list(self.topics.get(topic, []))
        for cb in callbacks:
            cb(data)

    # --- Services (Request/Reply) ---
    def register_service(self, service_name, handler_function):
        """Registers a function that will be executed when called."""
        with self.lock:
            self.services[service_name] = handler_function

    def call_service(self, service_name, request_data=None, timeout=10.0):
        """
        Calls the handler. If execution exceeds `timeout` (in seconds),
        returns None instead of raising an exception.
        """
        with self.lock:
            handler = self.services.get(service_name)
            
        if not handler:
            raise ValueError(f"Service '{service_name}' not found!")

        future = self.executor.submit(handler, request_data)
        
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            return None