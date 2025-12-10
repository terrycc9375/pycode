
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
    
    import brutal, numpy
    matrix = numpy.array([3,2,4,9,9,6,4,1,1,2,9,9,8,3,6,5,4,7,5,4,3,7,7,4,3,7,7,9,6,3,5,6,6,6,8,7,4,5,4,1,9,3,5,8,8,7,8,6,8,6,4,2,2,9,8,4,2,9,9,2,1,6,3,7,6,7,5,5,6,6,4,8,8,4,7,2,8,3,2,9,5,4,2,1,5,1,1,1,1,2,3,9,3,9,9,1,5,2,5,3,8,1,8,5,6,5,7,5,3,9,1,1]).reshape((14, 8))
    mat = brutal.eliminate(matrix, 2, 2, 3, 4)
    print(mat)

if __name__ == "__main__":
    main()

