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

def main():
    # prepare the input image
    os.makedirs("./temp", exist_ok=True)
    image_source = g2m.to_png("./temp", "temp.png")
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
    calculate(matrix)

def build_prefix(matrix: numpy.ndarray):
    rows, cols = matrix.shape
    prefix = numpy.zeros((rows+1, cols+1), dtype=int)
    for i in range(1, rows+1):
        for j in range(1, cols+1):
            prefix[i, j] = prefix[i-1, j] - prefix[i-1, j-1] + prefix[i, j-1] + matrix[i-1, j-1]
    return prefix

def sub_sum(prefix, r1, r2, c1, c2):
    return prefix[r2+1, c2+1] - prefix[r2+1, c2]- prefix[r1, c2+1] + prefix[r1, c1]

def eliminate(matrix, r1, r2, c1, c2):
    matrix[r1:r2+1, c1:c2+1] = 0
    return

def find_rect(matrix, prefix):
    rows, cols = matrix.shape
    rects = list()
    for r1 in range(rows):
        for r2 in range(r1, rows):
            for c1 in range(cols):
                for c2 in range(c1, cols):
                    s = sub_sum(prefix=prefix, r1=r1, r2=r2, c1=c1, c2=c2)
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    if s == 10 and area > 1:
                        non_zero_count = numpy.sum(matrix[r1:r2+1, c1:c2+1] != 0)
                        rects.append((non_zero_count, area, r1, r2, c1, c2))
    rects.sort(reverse=True)
    return rects

def calculate(matrix: numpy.ndarray):
    score = 0
    prefix = build_prefix(matrix)
    while True:
        rects = find_rect(matrix=matrix, prefix=prefix)
        if not rects:
            print(score)
            return
        non_zero, area, r1, r2, c1, c2 = rects[0]
        score += non_zero
        eliminate(matrix, r1, r2, c1, c2)
        prefix = build_prefix(matrix)
        with open("./log.txt", 'a+') as ouf:
            ouf.write(matrix.__str__())
            ouf.write("\n\n")



if __name__ == "__main__":
    main()
