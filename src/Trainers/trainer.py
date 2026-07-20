

class Trainer:
    def __init__(self):
        self.bag = None
        self.model = None
        self.history, self.loss, self.accuracy = None, None, None
        self.result = None

        self.PLOTS_DIRECTORY = "../plots/"