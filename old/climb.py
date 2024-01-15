from old.hold import Hold

class Climb:
    def __init__(self, holds, benchmark=None, name=None, grade=None):
        self.grade = grade
        if isinstance(holds[0], str):
            self.holds = [Hold(hold[0], hold[1:]) for hold in holds]
        else:
            self.holds = [Hold(hold["Description"][0], hold["Description"][1:]) for hold in holds]
        self.benchmark = benchmark
        self.name = name
    def to_vector(self):
        climb_vector = [0 for _ in range(18 * 11)]
        for hold in self.holds:
            climb_vector[11 * hold.row + hold.col] = 1
        return climb_vector