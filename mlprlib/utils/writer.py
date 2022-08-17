class Writer:
    """File Writer helper to avoid repeating code"""
    def __init__(self, filename: str):
        self.f = open(filename, "w")

    def write(self, line: str):
        self.f.write(line + '\n')

    def destroy(self):
        self.f.close()

    def __call__(self, line: str):
        self.write(line)
