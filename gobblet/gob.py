import numpy as np

class board():
    def __init__(self):
        #player1 is 1, player 2 is -1
        #4 is big, 1 is small
        self.p1 = [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
        self.p2 = [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
        self.player = 1
        self.current_board = self.init_board()




    def init_board(self):
        board = []
        for i in range(0,4):
            row  = []
            for j in range(0,4):
                row.append([])
            board.append(row)
        return board

    def reset(self):
        self.player = 1
        self.p1 = [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
        self.p2 = [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
        self.current_board = self.init_board()

    def change_player(self):
        self.player = self.player * -1

    def three_in_row(self, row, col):
        rsum = 0
        csum = 0
        dsum = 0
        sign = self.player * -1
        board = self.current_board
        for r in board:
            for c in board:
                if c != []:
                    c = c* sign

        for i in range(0,4):
            loc = board[row][i]
            if len(loc) != 0 and loc[-1] > 0:
                rsum += 1

            loc = board[i][col]
            if len(loc) != 0 and loc[-1] > 0:
                csum += 1

            loc = board[i][i]
            if row == col and len(loc) != 0 and loc[-1] > 0:
                dsum +=1
            loc = board[i][3-i]
            if row == 3 - col and len(loc) != 0 and loc[-1] > 0:
                dsum +=1

        if rsum == 3 or csum == 3 or dsum == 3:
            return True
        return False

    def is_opponent(self, row, col):
        sign = self.player * -1
        loc = self.current_board[row][col]
        if len(loc) == 0:
            return False
        if loc[-1]/loc[-1] == self.player:
            return False
        return True


    def valid_action(self, stack, p_row, p_col, row, col):
        pieces =  self.p1
        start_loc = self.current_board[p_row][p_col]
        end_loc = self.current_board[row][col]
        if self.player == -1:
            pieces = self.p2
        if stack != -1:
            if len(pieces[stack]) == 0:
                print("No pieces in that stack")
                return False
            if len(end_loc) != 0 and abs(end_loc[-1]) > pieces[stack][-1]:
                print("Can't gobble a bigger piece")
                return False
            if self.is_opponent(row, col) and not self.three_in_row(row,col):
                print("Can't gobble unless 3 in a row")
                return False
            return True

        if len(start_loc) == 0:
            print("No piece to move")
            return False

        if len(end_loc) != 0:
            if abs(start_loc[-1]) < abs(end_loc[-1]):
                print("Can't gobble a bigger piece")
                return False
        return True


    def place_piece(self, stack, prow, pcol, row, col):
        piece = 0
        if self.valid_action(stack, prow, pcol, row, col):
            if stack != -1:
                if self.player == 1:
                    piece = self.p1[stack].pop()
                else:
                    piece = -1 * self.p2[stack].pop()
            else:
                piece = self.current_board[prow][pcol].pop()
        else:
            print("Not Valid")
            return
        self.current_board[row][col].append(piece)

    def print_board(self):
        for i in range(0, 4):
            row = []
            for j in range(0, 4):
                loc = self.current_board[i][j]
                if len(loc) == 0:
                    row.append("none")
                else:
                    row.append(str(loc[-1]))
            print(row)

    #-1 means player2 won, 1 means player1 won, 0 means no win
    def won(self):
        #check rows for winner
        for i in range(0,4):
            sum = 0
            loc = self.current_board[i][0]
            if len(loc) == 0:
                continue
            sign = loc[-1]/abs(loc[-1])
            for j in range(0,4):
                loc = self.current_board[i][j]
                if len(loc) == 0:
                    break
                if loc[-1]/abs(loc[-1]) == sign:
                    sum += 1
                else:
                    break
            if sum == 4:
                if sign == 1:
                    print("player 1 wins")
                else:
                    print("player 2 wins")
                return True

        #check columns for winner
        for i in range(0,4):
            sum = 0
            loc = self.current_board[0][i]
            if len(loc) == 0:
                continue
            sign = loc[-1]/abs(loc[-1])
            for j in range(0,4):
                loc = self.current_board[j][i]
                if len(loc) == 0:
                    break
                if loc[-1]/abs(loc[-1]) == sign:
                    sum += 1
                else:
                    break
            if sum == 4:
                if sign == 1:
                    print("player 1 wins")
                else:
                    print("player 2 wins")
                return True

        #check left diagonal
        loc = self.current_board[0][0]
        if len(loc) != 0:
            sum = 0
            sign = loc[-1]/abs(loc[-1])
            for i in range(0,4):
                loc = self.current_board[i][i]
                if len(loc) == 0:
                    break
                if sign == loc[-1]/abs(loc[-1]):
                    sum +=1
            if sum == 4:
                if sign == 1:
                    print("player 1 wins")
                else:
                    print("player 2 wins")
                return True

        #check right diagonal
        loc = self.current_board[0][3]
        if len(loc) != 0:
            sum = 0
            sign = loc[-1]/abs(loc[-1])
            for i in range(0,4):
                loc = self.current_board[i][3-i]
                if len(loc) == 0:
                    break
                if sign == loc[-1]/abs(loc[-1]):
                    sum +=1
            if sum == 4:
                if sign == 1:
                    print("player 1 wins")
                else:
                    print("player 2 wins")
                return True
        # print("No winners")
        return False



g = board()
print("-1 for stack means moving a piece on the board. 0,1,2 represent each stack")
print("start row and start col ignored if you're moving from a stack")
while True:
    print("p1 pieces: ", g.p1)
    print("p2 pieces: ", g.p2)
    value = input("stack #, start row, start col, end row, end col\n")
    if(value == 'r'):
        g.reset()
        continue
    tmp = value.split(",")
    if(len(tmp) != 5):
        print("not enough inputs")
        continue
    stack = int(tmp[0])
    prow = int(tmp[1])
    pcol = int(tmp[2])
    row = int(tmp[3])
    col = int(tmp[4])

    g.place_piece(stack, prow, pcol, row, col)
    g.print_board()
    g.won()
    # g.change_player()
