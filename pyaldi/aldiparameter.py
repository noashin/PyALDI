class ALDIParameter:
    def __init__(self, init_val):
        self.value = init_val

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def get_energy_grad(self, *args):
        raise NotImplementedError

    def get_energy_for_value(self, value, *args):
        raise NotImplementedError

    def get_energy(self, *args):
        raise NotImplementedError
