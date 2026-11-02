class System:
    def __init__(self):
        self.users = {}
        self.current_user = None

    def register(self, name, password):
        if name in self.users:
            return False

        self.users[name] = password
        return True

    def login(self, name, password):
        if name in self.users and self.users[name] == password:
            self.current_user = name
            return True

        return False
        
    def get_profile(self):
        if self.current_user is None:
            raise PermissionError

        return {"name": self.current_user}

    def logout(self):
        self.current_user = None
        return True