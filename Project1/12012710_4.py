import numpy as np
import random
import time
import math
import numba

# from numba import jit
infinity = 1000000000000

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
COLOR_USABLE = 2
random.seed(0)

whiteCnt = 2
blackCnt = 2
corner = [(0, 0), (0, 7), (7, 0), (7, 7)]


# class AI(object):
#     # chessboard_size, color, time_out passed from agent
#     def __init__(self, chessboard_size, color):
#         self.chessboard_size = chessboard_size
#         # You are white or black
#         self.color = color
#         self.enemy_color = 0 - color
#         # the max time you should use, your algorithm's run time must not exceed the time limit.
#
#         # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
#         self.candidate_list = []
#         # The input is the current chessboard. Chessboard is a numpy array.
#         self.coner = [(0, 0), (0, chessboard_size - 1), (chessboard_size - 1, 0), (chessboard_size - 1, chessboard_size - 1)]

class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        self.enemy_color = 0 - color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need to add your decision to your candidate_list. The system will get the end of your candidate_list as your decision.
        self.candidate_list = []
        # The input is the current chessboard. Chessboard is a numpy array.

    def go(self, chessboard):
        emptyCnt = self.chessboard_size ** 2 - whiteCnt - blackCnt
        selfCnt = whiteCnt if self.color == 1 else blackCnt
        enemyCnt = 0 - selfCnt

        def miniMax(self, depth, chessboard):

            def max_value(depth, chessboard, alpha, beta):
                position_possible = goList(chessboard, self.color)

                if (emptyCnt == 0 and enemyCnt < selfCnt): return -infinity, None
                if (emptyCnt == 0 and enemyCnt > selfCnt): return infinity, None

                if depth <= 0 or not position_possible:
                    return getScore_chessboard(chessboard, self.color), None

                v, move = -infinity, None

                for position in position_possible:
                    if alpha > beta: break
                    chessboard_d = chessboard.copy()
                    changeChess(chessboard_d, position, self.color)
                    v2, _ = min_value(depth - 1, chessboard_d, alpha, beta)
                    if v2 > alpha: alpha = v2
                    if v2 > v:
                        v = v2
                        move = position
                return v, move

            def min_value(depth, chessboard, alpha, beta):
                position_possible = goList(chessboard, self.enemy_color)

                if (emptyCnt == 0 and enemyCnt < selfCnt): return infinity, None
                if (emptyCnt == 0 and enemyCnt > selfCnt): return -infinity, None

                if depth <= 0 or not position_possible:
                    return getScore_chessboard(chessboard, self.color), None

                v, move = infinity, None

                for position in position_possible:
                    if alpha > beta: break
                    chessboard_d = chessboard.copy()
                    changeChess(chessboard_d, position, self.enemy_color)
                    v2, _ = max_value(depth - 1, chessboard_d, alpha, beta)
                    if v2 < beta: beta = v2
                    if v2 < v: v = v2
                return v, move

            return max_value(depth, chessboard, -infinity, infinity)

        self.candidate_list = goList(chessboard, self.color)

        if (emptyCnt > 5):
            action_point = miniMax(self, 3, chessboard)[1]
        else:
            action_point = miniMax(self, infinity, chessboard)[1]

        if action_point in self.candidate_list:
            self.candidate_list.remove(action_point)
            self.candidate_list.append(action_point)
        return self.candidate_list


chessboard_score = np.array([[10000, -200, 250, 250, 250, 250, -200, 10000],
                                [-200, -200, -1, -1, -1, -1, -200, -200],
                                [-25, -1, 3, 2, 2, 3, -1, -25],
                                [-25, -1, 2, 1, 1, 2, -1, -25],
                                [-100, -1, 2, 1, 1, 2, -1, -100],
                                [-200, -100, 3, 2, 2, 3, -100, -200],
                                [-200, -200, -100, -1, -1, -100, -200, -200],
                                [10000, -200, -200, -100, -100, -200, -200, 10000]])


def getScore_chessboard(chessboard, color):
    w1 = 10  # chesscnt_diff
    w2 = 5  # chessboard_score
    w3 = 1

    chessboard_size = len(chessboard)
    enemy_color = 0 - color
    selfCnt = whiteCnt if color == 1 else blackCnt
    enemyCnt = 0 - selfCnt

    def getScore_score_diff():
        selfScore = 0
        enemyScore = 0
        for i in range(chessboard_size):
            for j in range(chessboard_size):
                if chessboard[i, j] == color:
                    selfScore += chessboard_score[i, j]
                if chessboard[i, j] == enemy_color:
                    enemyScore += chessboard_score[i, j]
        return selfScore - enemyScore

    def getScore_action():
        # self_action = goList(chessboard, color)
        enemy_action = goList(chessboard, enemy_color)
        return len(enemy_action)

    def getScore_aliveChess():
        return selfCnt - enemyCnt

    score_diff = getScore_score_diff()
    action_diff = getScore_action()
    cnt_diff = getScore_aliveChess()
    # diff = getScore_diff()

    return -(score_diff * w1 + action_diff * w2 + cnt_diff * w3)


def move(i, j, act):
    if act == 0:
        return i, j + 1  # right
    elif act == 1:
        return i, j - 1  # left
    elif act == 2:
        return i - 1, j  # up
    elif act == 3:
        return i + 1, j  # down
    elif act == 4:
        return i - 1, j + 1  # up right
    elif act == 5:
        return i + 1, j + 1  # down right
    elif act == 6:
        return i - 1, j - 1  # up left
    elif act == 7:
        return i + 1, j - 1  # down left


# @jit(nopython=True)
def changeChess(chessboard, point, color):
    chessboard[point] = color

    def doChange(point1, point2):
        global whiteCnt
        global blackCnt
        (i, j) = point1
        (a, b) = (round((point2[0] - point1[0]) / abs(point2[0] - point1[0] + 1e-5)),
                  round((point2[1] - point1[1]) / abs(point2[1] - point1[1] + 1e-5)))
        while (i, j) != point2:
            chessboard[(i, j)] = color
            whiteCnt += color
            blackCnt -= color
            i += a
            j += b

    possible_position = find(chessboard, point, color)
    for position in possible_position:
        if chessboard[position] == color:
            doChange(point, position)


def inSize(i, j, size):
    return 0 <= i < size and 0 <= j < size


def find(chessboard, position, color):
    enemyChess = 0 - color

    chessboard_size = len(chessboard)

    (x, y) = position

    xPossible = []
    yPossible = []

    (i, j) = (x, y)
    cnt = 0

    for act in range(8):
        i = x
        j = y
        cnt = 0
        i, j = move(i, j, act)
        while inSize(i, j, chessboard_size):
            if chessboard[i][j] != enemyChess:
                break
            i, j = move(i, j, act)
            cnt += 1
        if not (inSize(i, j, chessboard_size)):
            continue
        else:
            if cnt > 0:
                xPossible.append(i)
                yPossible.append(j)
    return list(zip(xPossible, yPossible))


# @jit(nopython=True)
def goList(chessboard, color):
    selfChess = color
    enemyChess = 0 - color

    def update(chessboard):  # 返回一个np.array,其中可用格子对应的值为2
        idx = chessboard.copy()
        myChess = np.where(chessboard == selfChess)
        myChess = list(zip(myChess[0], myChess[1]))
        for position in myChess:
            usableList = find(chessboard, position, selfChess)
            for gridPos in usableList:
                if idx[gridPos[0]][gridPos[1]] == 0:
                    idx[gridPos[0]][gridPos[1]] = 2

        return idx

    idx = np.where(update(chessboard) == COLOR_USABLE)
    idx = list(zip(idx[0], idx[1]))
    idx_d = []
    for i in idx:
        if i not in idx_d:
            idx_d.append(i)
    return idx_d

# def check(chessboard, color):
#
#     cnt = 0
#     for i in range(len(chessboard)):
#         for j in range(len(chessboard[i])):
#             if chessboard[(i, j)] == color:
#                 cnt += 1
#
#     return cnt

# import numpy as np
# ai = AI(8, -1)
# chessboard = np.array([[0,1,0,0,1,0,1,0],[1,1,-1,-1,-1,-1,1,1],[0,1,1,-1,-1,-1,-1,-1],[1,1,1,-1,1,1,-1,1],[0,1,-1,1,1,-1,1,1],[0,1,1,-1,-1,1,1,1],[1,1,1,1,1,-1,1,1],[1,1,0,1,1,-1,1,0]])
# print(ai.go(chessboard))
