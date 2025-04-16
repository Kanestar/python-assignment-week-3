class Smartphone:
    def __init__(self, brand, model, storage, os):
        # Encapsulated attributes
        self._brand = brand  # Protected attribute
        self._model = model
        self._storage = storage  # in GB
        self._os = os
        self.__imei = self.__generate_imei()  # Private attribute
    
    def __generate_imei(self):
        """Private method to generate a fake IMEI"""
        import random
        return ''.join([str(random.randint(0, 9)) for _ in range(15)])
    
    def display_info(self):
        return f"{self._brand} {self._model} ({self._storage}GB, {self._os})"
    
    def upgrade_os(self, new_os):
        self._os = new_os
        return f"OS upgraded to {new_os}"
    
    def get_imei(self):
        return self.__imei[-4:]  # Only show last 4 digits for security


# Inheritance example
class FoldableSmartphone(Smartphone):
    def __init__(self, brand, model, storage, os, unfolded_size):
        super().__init__(brand, model, storage, os)
        self.unfolded_size = unfolded_size
        self.__is_folded = True
    
    def toggle_fold(self):
        self.__is_folded = not self.__is_folded
        state = "folded" if self.__is_folded else "unfolded"
        return f"Phone is now {state}"
    
    def display_info(self):  # Method overriding
        return f"{super().display_info()}, Unfolded: {self.unfolded_size}\""


# Usage
phone1 = Smartphone("Apple", "iPhone 15", 128, "iOS")
print(phone1.display_info())
print(phone1.upgrade_os("iOS 17"))

foldable = FoldableSmartphone("Samsung", "Galaxy Z Fold", 256, "Android", 7.6)
print(foldable.display_info())
print(foldable.toggle_fold())
print(f"IMEI (last 4): {phone1.get_imei()}")