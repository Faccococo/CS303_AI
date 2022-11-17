import numpy as np
import random
import time
import math
from threading import Thread
infinity = math.inf

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
COLOR_USABLE = 2
random.seed(0)

class MyThread(Thread):

    #线程函数来自https://zhuanlan.zhihu.com/p/91601448
    def __init__(self, func, args):
        '''
        :param func: 可调用的对象
        :param args: 可调用对象的参数
        '''
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def getResult(self):
        return self.result


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

        def miniMax(self, depth, chessboard):
            
            def max_value(depth, chessboard, alpha, beta):
                position_possible = goList(chessboard, self.color)

                if depth <= 0 or not position_possible:
                    return getScore_chessboard(chessboard, self.color)
                v, move = -infinity, None

                for position in position_possible:
                    if alpha > beta : break
                    chessboard_d = chessboard.copy()
                    change_cnt = changeChess(chessboard_d, position, self.color)
                    v2, _ = min_value(depth - 1, chessboard_d, alpha, beta)
                    v2 += getScore_step(chessboard, position, self.color, change_cnt)
                    if v2 > alpha: alpha = v2
                    if v2 > v:
                        v = v2
                        move = position
                return v, move

            def min_value(depth, chessboard, alpha, beta):
                position_possible = goList(chessboard, self.enemy_color)

                if depth <= 0 or not position_possible:
                    return getScore_chessboard(chessboard, self.enemy_color)
                v, move = infinity, None

                for position in position_possible:
                    if alpha > beta : break
                    chessboard_d = chessboard.copy()
                    change_cnt = changeChess(chessboard_d, position, self.enemy_color)
                    v2, _ = max_value(depth - 1, chessboard_d, alpha, beta)
                    v2 += getScore_step(chessboard, position, self.enemy_color, change_cnt)
                    if v2 < beta: beta = v2
                    if v2 < v: v = v2
                return v, move

            return max_value(depth, chessboard, -infinity, infinity)

        self.candidate_list = goList(chessboard, self.color)
        self.candidate_list.append(miniMax(self, 4, chessboard))
        return self.candidate_list

def getScore_chessboard(chessboard, color):

    w1 = 1

    chessboard_size = len(chessboard)
    enemy_color = 0 - color

    def getScore_diff():
        chessboard_score = np.array([   [500,-25, 10,  5,  5, 10,-25,500],
                                        [-25,-45,  1,  1,  1,  1,-45,-25],
                                        [ 10,  1,  3,  2,  2,  3,  1, 10],
                                        [  5,  1,  2,  1,  1,  2,  1,  5],
                                        [  5,  1,  2,  1,  1,  2,  1,  5],
                                        [ 10,  1,  3,  2,  2,  3,  1, 10],
                                        [-25,-45,  1,  1,  1,  1,-45,-25],
                                        [500,-25, 10,  5,  5, 10,-25,500]  ])
        selfCount = 0
        selfScore = 0
        enemyCount = 0
        enemyScore = 0
        for i in range(chessboard_size):
            for j in range(chessboard_size):
                if chessboard[i, j] == color:
                    selfCount += 1
                    selfScore += chessboard_score[i, j]
                if chessboard[i, j] == enemy_color:
                    enemyCount += 1
                    enemyScore += chessboard_score[i, j]
        return 0 - ((selfCount - enemyCount) * 5 + (selfScore - enemyScore))

    t1 = MyThread(getScore_diff, ())
    t1.start()
    t1.join()
    diff = t1.getResult()
    # diff = getScore_diff()

    return w1 * diff

def getScore_step(chessboard, point, color, chess_change):
    enemyColor = 0 - color

    #Weight
    w1 = 1
    w2 = 10

    chessboard_score = np.array([   [500,-25, 10,  5,  5, 10,-25,500],
                                    [-25,-45,  1,  1,  1,  1,-45,-25],
                                    [ 10,  1,  3,  2,  2,  3,  1, 10],
                                    [  5,  1,  2,  1,  1,  2,  1,  5],
                                    [  5,  1,  2,  1,  1,  2,  1,  5],
                                    [ 10,  1,  3,  2,  2,  3,  1, 10],
                                    [-25,-45,  1,  1,  1,  1,-45,-25],
                                    [500,-25, 10,  5,  5, 10,-25,500]  ])

    #functions
    def getScore_chessValue():
        return 0 - chessboard_score[point[0], point[1]]

    #multi-tread
    t1 = MyThread(getScore_chessValue, ())
    t1.start()
    t1.join()
    chessValue = t1.getResult()
    # chessValue = getScore_chessValue()
    # chessChange = getScore_chessChange()

    return 0 - (w1 * chessValue + w2 * chess_change)

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

def changeChess(chessboard, point, color):
    chessboard[point] = color
    change_cnt = 0

    #point1 origin, point2 destination
    #from point1 to point2
    def doChange(point1, point2):
        change_cnt = 0
        (i, j) = point1
        (a, b) = (round((point2[0] - point1[0]) / abs(point2[0] - point1[0] + 1e-5)), 
                    round((point2[1] - point1[1]) / abs(point2[1] - point1[1] + 1e-5)))
        while (i, j) != point2:
            chessboard[(i, j)] = color
            change_cnt += 1
            i += a
            j += b
        return change_cnt
    
    possible_position = find(chessboard, point, color)
    for position in possible_position:
        if chessboard[position] == color:
            change_cnt += doChange(point, position)
    return change_cnt

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


import numpy as np
chessboard = np.array([ [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0,-1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]])

ai = AI()







































































