import numpy
import typing
from g2m import SIZE_X, SIZE_Y

class Rect:
    def __init__(self, start_x: int, start_y: int, width: int, height: int):
        self.start_x = start_x
        self.start_y = start_y
        self.width = width
        self.height = height
    def __repr__(self):
        return f"Rect({self.start_x}, {self.start_y}, {self.width}x{self.height})"

def eliminate(matrix: numpy.ndarray, rect: Rect | None) -> tuple[numpy.ndarray, int]:
    """
    erase target chunk to 0 and return erased count
    """
    new_matrix = matrix.copy()
    score = 0
    if rect is not None:
        x, y, w, h = rect.start_x, rect.start_y, rect.width, rect.height
        new_matrix[y:(y + h), x:(x + w)] = 0
        score = numpy.count_nonzero(matrix[y:(y + h), x:(x + w)])
    return new_matrix, score # type: ignore

def generate(matrix: numpy.ndarray) -> list[Rect]:
    """
    find all rects in the input matrix
    """
    result: list[Rect] = list()
    for r1 in range(0, SIZE_Y):
        for r2 in range(r1, SIZE_Y):
            for c1 in range(0, SIZE_X):
                for c2 in range(c1, SIZE_X):
                    if numpy.sum(matrix[r1:(r2+1), c1:(c2+1)]) == 10:
                        result.append(Rect(c1, r1, c2 - c1 + 1, r2 - r1 + 1))
    return result

class Node:
    def __init__(self, parent_matrix: numpy.ndarray, choice: Rect | None, parent: Node | None): # type: ignore
        self.matrix, self.score = eliminate(parent_matrix, choice)
        self.parent = parent
        self.choice = choice
        self.children: list[Node] = list()
        self.best_steps: list[Rect | None]
        self.best_score = 0
        self.score += self.parent.score if self.parent is not None else 0

    def apply(self, rect: Rect) -> 'Node':
        new_matrix = eliminate(self.matrix, rect)
        child = Node(matrix=new_matrix, choice=rect, parent=self)
        self.children.append(child)
        return child
    
    def get_path(self) -> list[Rect]:
        path = list()
        current = self
        while current.choice is not None:
            path.append(current.choice)
            current = current.parent
        return path[::-1]
    
    def dfs(self) -> tuple[list[Rect | None], int]:
        """
        1. scan the matrix and record possible steps.
        2.1 if no possible step, return score and steps.
        2.2 if there are some possible steps, dfs them, and return the best score of them.
        """
        possible_steps = generate(self.matrix)
        for step in possible_steps:
            self.children.append(Node(self.matrix, step, self))
        
        for child in self.children:
            child_steps, child_score = child.dfs()
            if child_score > self.best_score:
                self.best_score = child_score
                self.best_steps = child_steps
        self.best_steps.append(self.choice)
        
        return self.best_steps, self.best_score

