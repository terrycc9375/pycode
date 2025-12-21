import numpy
import typing
from NIKKE_OCR.model import SIZE_X, SIZE_Y

class Rect:
    def __init__(self, start_x: int, start_y: int, width: int, height: int):
        self.start_x = start_x
        self.start_y = start_y
        self.width = width
        self.height = height

    def __str__(self):
        return f"from ({self.start_y}, {self.start_x}) to ({self.start_y + self.height - 1}, {self.start_x + self.width - 1})"
    
    def __repr__(self):
        return f"Rect({self.start_x}, {self.start_y}, {self.width}x{self.height})"

memo: dict[bytes, tuple[list[Rect | None], int]] = {}

def eliminate(matrix: numpy.ndarray, rect: Rect | None) -> tuple[numpy.ndarray, int]:
    """
    erase target chunk to 0 and return erased count
    """
    new_matrix = matrix.copy()
    gain = 0
    if rect is not None:
        x, y, w, h = rect.start_x, rect.start_y, rect.width, rect.height
        new_matrix[y:(y + h), x:(x + w)] = 0
        gain = numpy.count_nonzero(matrix[y:(y + h), x:(x + w)])
    return new_matrix, gain # type: ignore

def generate(matrix: numpy.ndarray) -> list[Rect]:
    """
    find all rects in the input matrix
    """
    result: list[Rect] = list()
    for r1 in range(0, SIZE_Y):
        for r2 in range(r1, SIZE_Y):
            for c1 in range(0, SIZE_X):
                for c2 in range(c1, SIZE_X):
                    if r1 == r2 and c1 == c2:
                        continue
                    if numpy.sum(matrix[r1:(r2+1), c1:(c2+1)]) == 10:
                        result.append(Rect(c1, r1, c2 - c1 + 1, r2 - r1 + 1))
    return result

class Node:
    def __init__(self, parent_matrix: numpy.ndarray, choice: Rect | None, parent: "Node" = None): # type: ignore
        self.matrix, self.gain = eliminate(parent_matrix, choice)
        self.parent = parent
        self.choice = choice
        self.depth = 0
        self.children: list[Node] = list()
        self.best_child_score = 0
        self.best_child_steps: list[Rect | None] = list()
    
    def dfs(self) -> tuple[list[Rect | None], int]:
        """
        1. scan the matrix and record possible steps.
        2.1 if no possible step, return score and steps.
        2.2 if there are some possible steps, dfs them, and return the best score of them.
        """
        key = self.matrix.tobytes()
        if key in memo:
            return memo[key]
        
        possible_steps = generate(self.matrix)
        if self.parent:
            self.depth = self.parent.depth + 1
        with open("./log.txt", 'a') as f:
            f.write(f"Depth: {self.depth}, choices: {len(possible_steps)}\n")
        if not possible_steps:
            memo[key] = ([self.choice], 0)
            return [self.choice], 0
        for step in possible_steps:
            child = Node(self.matrix, step, self)
            child_steps, child_score = child.dfs()
            if child_score > self.best_child_score:
                self.best_child_score = child_score
                self.best_child_steps = child_steps
        
        result = ([self.choice] + self.best_child_steps, self.gain + self.best_child_score)
        memo[key] = result
        return result

