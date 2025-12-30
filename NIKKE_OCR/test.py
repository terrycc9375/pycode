
def main():
    # num = list()
    # for i in range(1, 11):
    #     with open(f"./dataset/train/{i}.txt", 'r') as f:
    #         content = f.readlines()
    #         for line in content:
    #             num.extend(list(map(lambda x: int(x), line.strip('\n'))))

    # import numpy as np
    # record = np.array([0 for _ in range(9)])
    # for n in num:
    #     record[n - 1] += 1
    # print(record)
    
    # import brutal, numpy
    # matrix = numpy.array([3,2,4,9,9,6,4,1,1,2,9,9,8,3,6,5,4,7,5,4,3,7,7,4,3,7,7,9,6,3,5,6,6,6,8,7,4,5,4,1,9,3,5,8,8,7,8,6,8,6,4,2,2,9,8,4,2,9,9,2,1,6,3,7,6,7,5,5,6,6,4,8,8,4,7,2,8,3,2,9,5,4,2,1,5,1,1,1,1,2,3,9,3,9,9,1,5,2,5,3,8,1,8,5,6,5,7,5,3,9,1,1]).reshape((14, 8))
    # mat = brutal.eliminate(matrix, 2, 2, 3, 4)
    # print(mat)

    import PIL.ImageGrab, PIL.Image, PIL.ImageDraw
    image = PIL.ImageGrab.grabclipboard()
    if image is None or isinstance(image, list):
        print("❌ [bold #fc535c]There is nothing in the clipboard.[/]")
        return None
    
    image = image.resize((1920, 1080), PIL.Image.Resampling.LANCZOS)
    # image = PIL.Image.open("./temp/2.png")
    # draw = PIL.ImageDraw.Draw(image)
    # y = 245
    # yy = 1080 - 33
    # draw.line(
    #     [(0, yy), (1920, yy)],
    #     fill="#222cf0",
    #     width=3
    # )
    image = image.crop((700, 245, 1220, 1047))
    image.save("./temp/1.png")
    dx = 520 // SIZE_X # 50
    dy = 802 // SIZE_Y # 50
    for j in range(SIZE_Y):
        for i in range(SIZE_X):
            left = i * dx
            right = left + dx
            upper = j * dy
            lower = upper + dy

            unit = img.crop((left, upper, right, lower))
            unit.save(f"{path}/input/{(id - 1) * 112 + j * SIZE_X + i}.png")


if __name__ == "__main__":
    main()

