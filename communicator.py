# a way to comunicate between real and virtual sensors and actuators
import threading

class Communicator:
    """Simulates ROS Topics and Services."""
    def __init__(self):
        self.topics = {}
        self.services = {}
        self.lock = threading.Lock()

    # --- Topics (Publish/Subscribe) ---
    def subscribe(self, topic, callback):
        with self.lock:
            if topic not in self.topics:
                self.topics[topic] = []
            self.topics[topic].append(callback)

    def publish(self, topic, data):
        with self.lock:
            callbacks = self.topics.get(topic, [])
        for cb in callbacks:
            cb(data)

    # --- Services (Request/Reply) ---
    def register_service(self, service_name, handler_function):
        """Registers a function that will be executed when called."""
        with self.lock:
            self.services[service_name] = handler_function

    def call_service(self, service_name, request_data=None):
        """Pauses the calling thread until the handler returns data."""
        with self.lock:
            handler = self.services.get(service_name)
            
        if handler:
            # Execute the function and return its result
            return handler(request_data)
        else:
            raise ValueError(f"Service '{service_name}' not found!")