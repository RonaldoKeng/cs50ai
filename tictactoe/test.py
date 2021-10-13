from tictactoe import player, winner

X = "X"
O = "O"
EMPTY = None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Check if won
    if winner(board) is not None:
        return True
    
    # Check if tied
    contains_EMPTY = False

    for row in board:
        for col in row:
             if col == EMPTY:
                contains_EMPTY = True
        
    if not contains_EMPTY:
        return True
    
    # Not terminal board if not won and not tied
    return False

test1 = [[EMPTY, EMPTY, EMPTY],
            [EMPTY, X, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

print(terminal(test1))
print(winner(test1))
print(player(test1))