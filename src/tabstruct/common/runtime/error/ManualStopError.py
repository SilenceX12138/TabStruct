class ManualStopError(Exception):

    def __init__(self, message="Manually stop the process."):
        self.message = message
        super().__init__(self.message)
