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
    return new_matrix, score

def generate(matrix: numpy.ndarray) -> list[Rect]:
    """
    find all rects in the input matrix
    """
    result: list[Rect] = list()
    for r1 in range(0, SIZE_Y):
        for r2 in range(r1, SIZE_Y):
            for c1 in range(0, SIZE_X):
                for c2 in range(c1, SIZE_X):
                    pass

    return result

class Node:
    def __init__(self, parent_matrix: numpy.ndarray, choice: Rect | None, parent: 'Node' = None):
        self.matrix, self.score = eliminate(parent_matrix, choice)
        self.choice = choice
        self.parent = parent
        self.children: list[Node] = list()
        self.best_child: typing.Optional[Node] = None

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

