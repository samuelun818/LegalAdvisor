
from sklearn.preprocessing import LabelEncoder

class text_converter:
    def __init__(self):
        self.le = LabelEncoder()
        self.bagofcodes = []

    def max_code(self):
        return max(self.bagofcodes)

    def min_code(self):
        return min(self.bagofcodes)

    def fit(self, bag):
        self.bagofcodes = self.le.fit_transform(bag)

    def encode(self, words):
        codes = self.le.transform(words)
        return codes

    def decode(self, codes):
        words = self.le.inverse_transform(codes)
        return words