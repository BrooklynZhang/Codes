import numpy as np
import torch
import random

negative_reward = -5.0
discount_vector = 0.7
p_state = 0.04
up, down, left, right = 0, 1, 2, 3
device = "cpu"
# goal mark, 0, 0, 0        [0][1][left]
# mark, 0, 0, 0, 0          [1][0][up]
# 0, 0, 0, 0, 0
# 0, 0, 0, 0, mark          [3][4][down]
# 0, 0, 0, mark, goal       [4][3][right]
batch_size = 64
rewardlist = [[0, 1, left], [1, 0, up], [4, 3, right], [3, 4, down]]
model = torch.nn.Sequential(
        torch.nn.Linear(5 * 5 * 4, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1)
    ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def get_optimal_route1(table):
    table1 = x = np.zeros((5, 5, 4))
    while True:
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                for steps in range(table.shape[2]):
                    magicmax = -500
                    if steps == up and i > 0:
                        magicmax = np.max(table[i - 1][j])
                    elif steps == down and i < table.shape[0] - 1:
                        magicmax = np.max(table[i + 1][j])
                    elif steps == left and j > 0:
                        magicmax = np.max(table[i][j - 1])
                    elif steps == right and j < table.shape[1] - 1:
                        magicmax = np.max(table[i][j + 1])

                    if [i, j, steps] in rewardlist:
                        table1[i][j][steps] = discount_vector * magicmax
                    else:
                        table1[i][j][steps] = negative_reward + discount_vector * magicmax
        if np.allclose(table, table1):
            break
        else:
            table = table1
    return table1


def get_optimal_route2(table, rewards):
    for iteration in range(2000):
        i = torch.randint(0, 5, (64,), device=device, requires_grad=False)
        j = torch.randint(0, 5, (64,), device=device, requires_grad=False)
        steps = torch.randint(0, 4, (64,), device=device, requires_grad=False)
        seed = torch.zeros((64, 5 * 5 * 4), device=device)
        seed.scatter_(1, (i*20+j*4+steps).reshape(batch_size, 1), 1)
        preditcion = model(seed).reshape(64)
        magicmax = torch.full((64,), -500, device=device, requires_grad=False)
        for step in range(64):
            if steps[step] == up and i[step] > 0:
                magicmax[step] = torch.max(table[i[step] - 1][j[step]])
            elif steps[step] == down and i[step] < table.shape[0] - 1:
                magicmax[step] = torch.max(table[i[step] + 1][j[step]])
            elif steps[step] == left and j[step] > 0:
                magicmax[step] = torch.max(table[i[step]][j[step] - 1])
            elif steps[step] == right and j[step] < table.shape[1] - 1:
                magicmax[step] = torch.max(table[i[step]][j[step] + 1])

        loss = (preditcion - (rewards[i, j, steps].detach() + discount_vector * magicmax.detach())) ** 2
        if(iteration % 100 == 0):
            print("Loop of:", iteration, "Loss:", loss.sum().item())
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
        table[i, j, steps] = preditcion
    return table

def generate_table():
    table = np.zeros((5, 5, 4))
    for i in range(len(table)):
        for j in range(len(table[i])):
            for steps in range(len(table[i][j])):
                table[i][j][steps] = random.random() * -10
    return table


def question1():
    table = generate_table()
    print("initial table:")
    print(table)
    newtable = get_optimal_route1(table)
    print("table after optimal route: ")
    print(newtable)


def question2():
    table = generate_table()
    print("initial table:")
    print(table)
    table = torch.tensor(table, dtype=torch.float, device=device)
    rewardstable = np.full((5, 5, 4), negative_reward, dtype=float)
    rewardstable[0][1][left], rewardstable[1][0][up], rewardstable[3][4][down], rewardstable[4][3][
        right] = 1, 1, 1, 1
    rewards = torch.tensor(rewardstable, dtype=torch.float, device=device, requires_grad=False)

    output = get_optimal_route2(table, rewards)
    print(output)


if __name__ == "__main__":
    print("Q1: ")
    question1()
    print("Q2: ")
    question2()
