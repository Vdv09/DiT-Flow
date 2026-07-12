class ClassRegistry:
    def __init__(self):
        self.classes = dict()
    
    def add_to_registry(self, key):
        def decorator(cls):
            self.classes[key] = cls
            return cls
        
        return decorator
    
    def get_from_registry(self, key, **kwargs):
        return self.classes[key](**kwargs)


class_registry = ClassRegistry()