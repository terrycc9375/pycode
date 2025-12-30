"""
model
1. CNN
    - Supervised learning for number recognition
    - 10 class (1~9 and blank is 0)
2. AlphaNum
    - Reinforcement learning for making optimal decision
    - Agent-Critic model
"""
import PIL
from PIL import Image, ImageGrab, ImageDraw
import time
import os
import rich, rich.console, rich.live, rich.panel
import torch
import torchvision
import numpy

SIZE_X = 10
SIZE_Y = 16

console = rich.console.Console()
def clipboard_to_png(out_dir="./dataset", filename="temp.png"):
    image = ImageGrab.grabclipboard()
    if image is None or isinstance(image, list):
        console.print("❌ [bold #fc535c]There is nothing in the clipboard.[/]")
        return None
    
    image = image.resize((635, 1080), Image.Resampling.LANCZOS)

    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, filename)
    try:
        image.save(output_path)
        console.print(f"[#2882c7]Photo size: {image.width} x {image.height}[/]")
    except Exception as e:
        console.print(f"[bold #ba1a32]Unknown error: {e}[/]")
        exit(1)
    return image

def resize(out_dir: str, filename: str):
    output_path = os.path.join(out_dir, filename)
    image = Image.open(output_path)
    image = image.resize((635, 1080), Image.Resampling.LANCZOS)
    image.save(output_path)
    print(f"{filename} saved.")
    return

def cut(path: str, id: int):
    os.makedirs(f"{path}/input", exist_ok=True)
    img = Image.open(f"{path}/{id}.png").convert("RGB")

    dx = 635 // SIZE_X
    dy = 1080 // SIZE_Y
    for j in range(SIZE_Y):
        for i in range(SIZE_X):
            left = i * dx
            right = left + dx
            upper = j * dy
            lower = upper + dy

            unit = img.crop((left, upper, right, lower))
            unit.save(f"{path}/input/{(id - 1) * 112 + j * SIZE_X + i}.png")


class DS(torch.utils.data.Dataset):
    def __init__(self, img_dir, label: list[int], transform=None):
        self.label = label
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index: int):
        image = Image.open(os.path.join(self.img_dir, f"{index}.png"))
        if self.transform is not None:
            image = self.transform(image)
        label = self.label[index] - 1
        return image, label

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool2d((7, 7)),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 * 7 * 7, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 9),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # flatten
        x = self.classifier(x)
        return x

def main():
    train_directory = "./dataset/train"
    default_img_name = "0.png"
    image_process = True
    MAX_EPOCH = 100
    pretrained = True

    label = list()
    for idx in range(10):
        id = idx + 1
        # 1. read image and crop
        if not image_process:
            cut(train_directory, id)
        # 2. read labels
        with open(f"{train_directory}/{id}.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = list(map(lambda x: int(x), line.strip(' ').strip('\n')))
                label.extend(item)
    
    # load data
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.ToTensor()
        ]
    )
    train_dataset = DS(f"{train_directory}/input", label, transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True)

    # load model
    if not pretrained:
        model = CNN().to("cuda")
    else:
        model = torch.load("model/001.pt", weights_only=False).to("cuda")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    console.print("[bold #16f251]Start to train[/]")
    with rich.live.Live(console=console) as live:
        start = time.time()
        eta = float(0)
        for epoch in range(MAX_EPOCH):
            for image, true_label in train_dataloader:
                image = image.to("cuda")
                true_label = true_label.to("cuda")
                optimizer.zero_grad()
                output = model(image)
                loss_function = torch.nn.CrossEntropyLoss()
                loss = loss_function(output, true_label)
                loss.backward()
                optimizer.step()

                body = f"Loss: {loss:.3f}, ETA: {eta:.2f}s"
                title = f"Epoch {epoch + 1}/{MAX_EPOCH}"
                panel = rich.panel.Panel(body, title=title, expand=False)
                live.update(panel)
            elapsed = time.time() - start
            eta = (MAX_EPOCH - epoch - 1) / (epoch + 1) * elapsed
            body = f"Loss: {loss:.3f}, ETA: {eta:.2f}s"
            title = f"Epoch {epoch + 1}/{MAX_EPOCH}"
            panel = rich.panel.Panel(body, title=title, expand=False)
            live.update(panel)

    #load test data
    test_directory = "./dataset/test"
    with open(f"{test_directory}/test.txt", 'r') as f:
        test_label = list()
        lines = f.readlines()
        for line in lines:
            item = list(map(lambda x: int(x), line.strip(' ').strip('\n')))
            test_label.extend(item)

    test_image = list()
    test_img_src = Image.open(f"{test_directory}/test.png").convert("RGB")
    dx = 635 // SIZE_X
    dy = 1080 // SIZE_Y
    for j in range(SIZE_Y):
        for i in range(SIZE_X):
            left = i * dx
            right = left + dx
            upper = j * dy
            lower = upper + dy
            test_image.append(test_img_src.crop((left, upper, right, lower)))

    # test
    model.eval()
    with torch.no_grad():
        correct = 0
        for index in range(len(test_label)):
            img = transform(test_image[index]).to("cuda")
            result = model(img.unsqueeze(0))
            prediction = result.argmax(dim=1).item() + 1
            if prediction == test_label[index]: correct += 1
            # else: print(f"Index: {index} | Prediction = {prediction}, Answer = {test_label[index]}")
        accuracy = correct / len(test_label)
        console.print(f"[#f2d116]Accuracy: {accuracy:.3f}[/]")

    os.makedirs("model", exist_ok=True)
    torch.save(model, "model/002.pt")


if __name__ == "__main__":
    main()
