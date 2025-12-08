"""
1. Screenshot
2. Run this code
3. This code shows the answer
4. 卷死他們
"""

import g2m
from g2m import CNN
import os
import torch, torchvision
import numpy
from PIL import Image, ImageDraw, ImageShow, ImageTransform

def main():
    # prepare the input image
    os.makedirs("./temp", exist_ok=True)
    image_source = g2m.to_png("./temp", "temp.png")
    if image_source is None:
        # load image from ./temp/temp.png
        image_source = Image.open("./temp/temp.png")
    image = list()
    dx = 635 // 8
    dy = 1080 // 14
    for j in range(14):
        for i in range(8):
            left = i * dx
            right = left + dx
            upper = j * dy
            lower = upper + dy
            image.append(image_source.crop((left, upper, right, lower)))
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor()
        ]
    )

    # load model and parse
    model = torch.load("model/002.pt", weights_only=False).to("cuda")
    model.eval()
    output_list = list()
    with torch.no_grad():
        for i in range(112):
            img = transform(image[i]).to("cuda")
            output = model(img.unsqueeze(0))
            number = output.argmax(dim=1).item() + 1
            output_list.append(number)
    matrix = numpy.array(output_list).reshape((14, 8))
    print(matrix)
    grade = dfs(matrix)

def eliminate(matrix, r1, r2, c1, c2):
    matrix[r1:r2+1, c1:c2+1] = 0
    return

def diffusion(matrix, r1, r2, c1, c2):
    top = left = 0
    bottom = right = 100
    for y in range(r1, r2 + 1):
        for x in range(c1, c2 + 1):
            if matrix[y, x] == 0:
                top = max(0, y - 1, top)
                left = max(0, x - 1, left)
                bottom = min(13, y + 1, bottom)
                right = min(7, x + 1, right)
    return (top, bottom, left, right)

def find_submatrix(matrix, top, bottom, left, right, score):
    result = {
        "is_find": False,
        "output": None,
    }
    for row in range(top, bottom + 1):
        for col in range(left, right + 1):
            for dy in reversed(range(0, bottom - top + 1)):
                for dx in reversed(range(0, right - left + 1)):
                    if numpy.sum(matrix[row:row+dy, col:col+dx]) == 10:
                        # set to 0 and update matrix
                        result["is_find"] = True
                        score += numpy.count_nonzero(matrix[row:row+dy, col:col+dx])
                        eliminate(matrix, row, row+dy, col, col+dx)
                        top, bottom, left, right = diffusion(matrix, top, bottom, left, right)
                        result["output"] = (matrix, top, bottom, left, right, score)
                        return result
    return result
                

def dfs(matrix: numpy.ndarray):
    score = 0
    # step 1: find submatrix since numbers are compact in the beginning
    top, bottom, left, right = 0, 0, 0, 0
    # [2, 1]
    for row in range(13):
        for col in range(8):
            if numpy.sum(matrix[row:row+1, col]) == 10:
                top = row
                bottom = row + 1
                left = right = col
                score += 2
    # [1, 2]
    for row in range(14):
        for col in range(7):
            if numpy.sum(matrix[row, col:col+1]) == 10:
                top = bottom = row
                left = col
                right = col + 1
                score += 2
            
    # step 2: eliminate the initial submatrix and diffuse 1 block
    eliminate(matrix, top, bottom, left, right)
    top, bottom, left, right = diffusion(matrix, top, bottom, left, right)
    
    # step 3: dfs the best sequence of slices
    while True:
        result = find_submatrix(matrix, top, bottom, left, right, score)
        if not result["is_find"]:
            # spread attention submatrix by 1 block
            
            return score


if __name__ == "__main__":
    main()
