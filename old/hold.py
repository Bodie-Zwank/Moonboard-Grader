class Hold:
    def __init__(self, letter, num):
        # 18 rows on moonboard; 0th row is top
        self.row = 18 - int(num)
        # A = 65 ascii; 0th column is left
        self.col = ord(letter) - 65