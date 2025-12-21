"""
1. Screenshot
2. Run this code
3. This code shows the answer
4. 卷死他們
"""

import NIKKE_OCR.model as model
from NIKKE_OCR.model import CNN
from brutal import Rect, Node, eliminate, generate
import os
import time
import torch, torchvision
import numpy
from PIL import Image, ImageDraw, ImageShow, ImageTransform
import rich, rich.live, rich.panel

def main():
    # prepare the input image
    os.makedirs("./temp", exist_ok=True)
    image_source = model.to_png("./temp", "temp.png")
    if image_source is None:
        # load image from ./temp/temp.png
        image_source = Image.open("./temp/temp.png")
    image = list()
    dx = 635 // model.SIZE_X
    dy = 1080 // model.SIZE_Y
    for j in range(model.SIZE_Y):
        for i in range(model.SIZE_X):
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
    model: CNN = torch.load("model/002.pt", weights_only=False).to("cuda")
    model.eval()
    output_list = list()
    with torch.no_grad():
        for i in range(112):
            img = transform(image[i]).to("cuda")
            output = model(img.unsqueeze(0))
            number = output.argmax(dim=1).item() + 1
            output_list.append(number)
    matrix = numpy.array(output_list).reshape((model.SIZE_Y, model.SIZE_X))
    root = Node(parent_matrix=matrix, choice=None, parent=None)

    print("start searching")
    console = rich.console.Console()
    start_time = time.time()
    with rich.live.Live(console=console, refresh_per_second=4) as live:
        elapsed = time.time() - start_time
        live.update(rich.panel.Panel(f"Elapsed: {elapsed:.1f}s", title=f"tracker", expand=False))
        best_steps, best_score = root.dfs()
    
    print(best_score)
    for i, rect in enumerate(best_steps, 1):
        with open("./result.txt", 'w') as f:
            f.write(f"Step {i:2d}: {rect}\n")
    # with open("./log.txt", 'w') as f:
    #     f.write(matrix.__str__() + '\n')
    #     for rect in generate(matrix):
    #         f.write(f"{rect}, mat = \n{matrix[rect.start_y:rect.start_y+rect.height, rect.start_x:rect.start_x+rect.width]}\n")


if __name__ == "__main__":
    main()
