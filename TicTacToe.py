import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import random
import json

#basic functions
def genBoard(dimension):
    table = [["-" for c in range(dimension)] for r in range(dimension)]
    return table

class TicTacToe:
    def __init__(self, dimension):
        self.dimension = dimension
        self.board_size = dimension**2
        self.board = np.zeros(self.board_size)
        self.actions = list(np.arange(dimension**2))
        self.possible_moves = np.arange(dimension**2)
        self.taken_moves = []
        self.current_state = []
        self.is_end_game = False

    def resetBoard(self):                       #set everyting to default
        self.board = np.zeros(self.board_size)
        self.possible_moves = np.arange(dimension**2)
        self.taken_moves = []
        self.current_state = []
        self.is_end_game = False

    def printBoard(self):
        # for i in range(self.dimension):
        #     for j in range(self.dimension):
        #         c = self.board[i][j]
        #         if(c == 1):
        #             print("X"),
        #         elif(c == -1):
        #             print("O"),
        #         else:
        #             print("-"),
                
        #     print("\n")
        print(self.board.reshape(self.dimension, self.dimension))

    def placeAction(self, ID, action):
        self.board[action] = ID #reflect move on tictactoe board
        self.taken_moves.append(action)         #stores actions taken to prevent from taking the same move
        rst = np.where(self.possible_moves == action)
        idx = int(rst[0])
        self.possible_moves = np.delete(self.possible_moves, idx)
        self.possible_moves = np.array(self.possible_moves)      #update possible moves
        self.current_state.append((ID*action))  #update current state
        return [self.getState(), self.getReward(ID), action]    #returns state, reward, and action
    
    def hasALine(self):
        size = self.dimension

        board = self.board.reshape(size, size)

        horizontal_sums = board.sum(axis=1)
        vertical_sums = board.sum(axis=0)
        diag_sum = np.trace(board)
        backDiag_sum = np.trace(np.fliplr(board))
        
        for i in range(self.dimension):
            h = abs(horizontal_sums[i])
            v = abs(vertical_sums[i])
            d = abs(diag_sum)
            b = abs(backDiag_sum)
            if (h == size or v == size or d == size or b == size):
                return True
        return False
    
    def isDraw(self):
        return np.count_nonzero(self.board) == (self.board_size) - 1
    
    def isEndGame(self):
        self.is_end_game = self.isDraw() or self.hasALine()
        return self.is_end_game

    def getReward(self, ID):
        if(self.hasALine()):
            return 100*ID
        elif(self.isDraw()):
            return 5
        else:
            return 0

    def getState(self):
        return np.sort(np.array(self.current_state))

    def getPossibleMoves(self):
        return self.possible_moves

    def getTakenMoves(self):
        return self.taken_moves

    def getBoardSize(self):
        return self.board_size

#for Q-learning
epsilon = 0.8

class Player():
    def __init__(self, ID, char, epsilon, learningRate, discountFactor):           #constructor
        self.ID = ID
        self.char = char
        self.q_table = []
        self.tree = []
        self.epsilon = epsilon
        self.learning_rate = learningRate
        self.discount_factor = discountFactor
        self.current_state = []
    
    def makeMove(self, tictactoe):
        if (tictactoe.isEndGame()):
            return tictactoe
        
        self.current_state = list(tictactoe.getState())[:]  #stores the old state
        
        if (self.current_state not in self.tree):
            action_q_values = []
            for i in range(tictactoe.getBoardSize()):
                action_q_values.append(1)
            self.tree.append(self.current_state[:])
            self.q_table.append(action_q_values)
        
        if np.random.random() < self.epsilon:                       #update the state, reward, and action taken for later calculation 
            qT = np.array(self.q_table[self.tree.index(self.current_state)])
            for i in range (len(qT)):
                if (i not in tictactoe.getPossibleMoves()):
                    qT[i] = -100
            act = np.argmax(qT)
            possible_acts = []
            for i in range(tictactoe.getBoardSize()):
                if qT[i] == qT[act]:
                    possible_acts.append(i)
            self.new_state, self.reward, self.action = tictactoe.placeAction(self.ID, random.choice(possible_acts))
        else:
            #print("randomly selected action")
            self.new_state, self.reward, self.action = tictactoe.placeAction(self.ID, random.choice(tictactoe.getPossibleMoves()))
        return tictactoe
    
    def makeCompMove(self, tictactoe):      #take random actions
        if (tictactoe.isEndGame()):
            return tictactoe
        self.new_state, self.reward, self.action = tictactoe.placeAction(self.ID, random.choice(tictactoe.getPossibleMoves()))
        return tictactoe

    def setEpsilon(self, e):
        self.epsilon = e
    
    def setLearningRate(self, lr):
        self.learning_rate = lr

    def setDiscountFactor(self, df):
        self.discount_factor = df
    
    def updateQ_Table(self):
        old_q_value = self.q_table[self.tree.index(self.current_state)][self.action]
        try:
            idx = self.tree.index(self.new_state)
            value = np.max(self.q_table[idx])
            possible_acts = []
            for i in range(tictactoe.getBoardSize()):
                if self.q_table[idx][i] == self.q_table[value]:
                    possible_acts.append(i)
            value = random.choice(possible_acts)
        except ValueError:
            value = 0
        temporal_difference = self.reward + (self.discount_factor * value) - old_q_value
        new_q_value = old_q_value + (self.learning_rate * temporal_difference)
        self.q_table[self.tree.index(self.current_state)][self.action] = new_q_value

    def getID(self):
        return self.ID

    def getCurrentState(self):
        return self.current_state

    
AI = Player(1, "X", epsilon, 0.4, 0.9)
NPC = Player(-1, "O", epsilon, 0.9, 0.9)

num = 100000

dimension = 3
tictactoe = TicTacToe(dimension)

odds = 0
draw = 0
winning_rate = []
draw_rate = []
lose_rate = []
x_axis = []

for episode in range (1, num+1, 1):
    if episode % 1000 == 0:
        NPC.q_table = AI.q_table
        odds = 0
        draw = 0
        lose = 0
        with open('qlearning.txt', 'w') as filehandle:
            json.dump(AI.q_table, filehandle)
        AI.setEpsilon(1.0)
        for j in range (1000):
            while not tictactoe.isEndGame():
                tictactoe = AI.makeMove(tictactoe)
                if (tictactoe.hasALine()):
                    odds += 1
                    break
                tictactoe = NPC.makeCompMove(tictactoe) #random move
                if (tictactoe.hasALine()):
                    lose += 1
                    break
                if tictactoe.isDraw():
                    draw += 1
            tictactoe.resetBoard()
        draw = (draw/(1000*1.0))*100
        odds = (odds/(1000*1.0))*100
        lose = (lose/(1000*1.0))*100
        winning_rate.append(odds)
        draw_rate.append(draw)
        lose_rate.append(lose)
        x_axis.append(episode)
        print("testing rates... odds = " + str(odds) + "% draw = " + str(draw) + "% lose = " + str(lose) + "%")
    AI.setEpsilon(epsilon)
    while not tictactoe.isEndGame():
        tictactoe = AI.makeMove(tictactoe)
        AI.updateQ_Table()
        if(tictactoe.hasALine()):
            #odds += 1
            #print("AI wins")
            break

        tictactoe = NPC.makeMove(tictactoe) #not random move
        # if(tictactoe.hasALine()):
        #     print("NPC wins")
        #NPC.updateQ_Table()
        epsilon += ((1-epsilon)*10)/num
        AI.setEpsilon(epsilon)
    # if tictactoe.isDraw():
    #     draw += 1
    tictactoe.resetBoard()
    print("game: " + str(episode) + " done")
    #print("\n")
    #winning_rate.append(odds)
#print("training completed, odd = " + str(odds) + "% draw = " + str(draw) + "%")
print("training completed")
print(max(winning_rate))

csfont = {'fontname':'Comic Sans MS'}

plt.plot(x_axis, winning_rate, label = "Win", linewidth = 3, color = "#A2F350")
plt.plot(x_axis, draw_rate, label = "Tie", linewidth = 3, color = "#1c4285")
plt.plot(x_axis, lose_rate, label = "Lose", linewidth = 3, color = "#F35050")
plt.xlabel("Episodes", **csfont)
plt.ylabel("Rates (%)", **csfont)
plt.title("Rates vs. Episodes", **csfont)
plt.legend(prop={'family':"Comic Sans MS"})
plt.show()



# with open('qlearning.npy', 'rb') as f:
#     a = np.load(f)

# print(a[15])