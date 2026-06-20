import matplotlib.pyplot as plt
import numpy as np


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def heuristic(p1, p2, kind="euclidean"):
    if kind == "euclidean":
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    elif kind == "manhattan":
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def astar(grid, start, end, weight=1.0):
    start_node = Node(None, start)
    end_node = Node(None, end)

    open_list = []
    closed_list = []
    open_list.append(start_node)

    movements = [(-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
                 (0, -1, 1.0), (0, 1, 1.0),
                 (1, -1, 1.414), (1, 0, 1.0), (1, 1, 1.414)]

    while open_list:
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        open_list.pop(current_index)
        closed_list.append(current_node)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1], closed_list

        for move in movements:
            node_position = (current_node.position[0] + move[0], current_node.position[1] + move[1])

            if (node_position[0] < 0 or node_position[0] >= grid.shape[0] or
                    node_position[1] < 0 or node_position[1] >= grid.shape[1]):
                continue

            if grid[node_position[0]][node_position[1]] != 0:
                continue

            new_node = Node(current_node, node_position)

            if any(closed_child for closed_child in closed_list if closed_child == new_node):
                continue

            new_node.g = current_node.g + move[2]
            new_node.h = weight * heuristic(new_node.position, end_node.position, "euclidean")
            new_node.f = new_node.g + new_node.h

            if any(open_node for open_node in open_list if new_node == open_node and new_node.g >= open_node.g):
                continue

            open_list.append(new_node)

    return None, closed_list


# ==================== 以下为新增的完整主程序调用与绘图逻辑 ====================
if __name__ == "__main__":
    # 1. 创建一个 20x20 的地图 (0代表空地，1代表障碍物)
    grid = np.zeros((20, 20))

    # 设置 3 道错开的障碍物墙
    grid[5, 2:19] = 1
    grid[10, 0:15] = 1
    grid[15, 5:20] = 1

    # 2. 定义起点和终点
    start = (0, 0)
    end = (19, 19)

    print("正在计算 A* 路径...")
    path, closed_nodes = astar(grid, start, end, weight=0.0)

    if path:
        print(f"规划成功！路径总长度(栅格代价): {len(path)}")

        # 3. 将路径坐标提取为 X 和 Y 用于绘图
        path_x = [p[1] for p in path]  # 注意：矩阵的行对应Y，列对应X
        path_y = [p[0] for p in path]

        # 4. 绘制地图与路径
        plt.figure(figsize=(8, 8))
        # cmap='binary' 让0显示为白色，1显示为黑色
        plt.imshow(grid, cmap='binary', origin='upper')

        # 画出规划出的最优路径（红线）
        plt.plot(path_x, path_y, color='red', linewidth=3, label="A* Path")

        # 标记起点和终点
        plt.scatter(start[1], start[0], color='green', s=200, marker='^', label="Start")
        plt.scatter(end[1], end[0], color='blue', s=200, marker='*', label="End")

        plt.title("A-Star Robot Path Planning")
        plt.legend()
        plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        plt.xticks(range(20))
        plt.yticks(range(20))

        print("正在弹出地图窗口，请查看...")
        plt.show()
    else:
        print("未找到有效路径，请检查障碍物是否封死了终点！")
