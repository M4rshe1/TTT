# M4rshe1/TTT

This is  Tic Tac Toe in PowerShell.  
The neural network is trained using python, TensorFlow and Keras, skit-learn and self generated data using the minimax algorithm.  
After that the neural network is saved and loaded in PowerShell to play against the user.


## Command

```
    TTT.ps1 [-player1 <string>] [-player2 <string>] [-FIRST_COMPUTER_MOVE_RANDOM] [-FIRST_NEURAL_NETWORK_MOVE_RANDOM] [-FIRST_MOVE_BY_PLAYER] [-help]

    -player1, -p1                           The symbol for player 1 (default: X)
    -player2, -p2                           The symbol for player 2 (default: O)
    -FIRST_COMPUTER_MOVE_RANDOM, -c         If set to true the computer will make the first move randomly (default: true)
    -FIRST_NEURAL_NETWORK_MOVE_RANDOM, -n   If set to true the neural network will make the first move randomly (default: true)
    -FIRST_MOVE_BY_PLAYER, -p               If set to true the player will make the first move (default: false)
                                            Will override FIRST_COMPUTER_MOVE_RANDOM and FIRST_NEURAL_NETWORK_MOVE_RANDOM to false
    -help, -h                               Show this help message
```