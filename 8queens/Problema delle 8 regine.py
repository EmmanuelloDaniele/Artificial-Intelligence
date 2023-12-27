def printBorad(board):
    for i in range(len(board)):
        for j in range(len(board)):
            print(board[i][j],end='')
        print()

    #Separatore solution/board
    separator = ''
    for i in range(len(board)):
        separator += '──'

    print (separator)

def conteggioSol(board,row):
    #definisco global per modificare il contatore al di fuori
    global counter

    #se tutte le regine vengono posizionate sulla board senza minacciarsi a vicenda
    #il che significa che anche la riga corrente è della stessa dimensione della board
    #quindi stampa la scacchiera con le queens e incrementa il contatore
    if row >= len(board):
        counter += 1
    
    for i in range(len(board)):

        #prima di posizionare una queen sulla borad controllo
        #che non ci siano altre queen esistenti che possano minacciare la queen attuale che sto per posizonare
        if isValid(board, row, i):

            #posiziona la queen[row][i]
            board[row][i] = 'Q'

            #ripeto per la riga successiva
            conteggioSol(board, row + 1)

            #Tornon indietro e rimuovo la regina successiva per trova la soluzione successiva
            board[row][i] = '·'

def placeQueen(board):
    row = 0
    column = 0
    while row in range(len(board)):
        while column in range(len(board)):
            if isValid(board, row, column):
                board[row][column] = 'Q'
                row += 1
                column = 0
                break
            elif column +1 == len(board):
                c = 0 
                while c in range(len(board)):
                    if board[row-1][c] == 'Q':
                        board[row-1][c] = '·'
                        if row > 0:
                            row -= 1
                            column = c + 1
                        else:
                            row = 0 
                            column = c + 1 
                        if column < 4:
                            break
                        else:
                            c = 0
                            continue
                    else:
                        c +=1
            else:
                column += 1
        if row >= len(board):
            printBorad(board)

def checkColumn(chessBoard, row, column):
    #controlla se due regina sono nella stessa colonna
    for r in range(row):
        if chessBoard[r][column] == 'Q':
            return False
        
    return True

def checkDiagonal(chessBoard,row,column):
    for r in range(len(chessBoard)):
        for c in range(len(chessBoard)):
            if(r + c == row + column) or (r - c == row - column):
                if chessBoard[r][c] == 'Q':
                    return False
                
    return True


def isValid(chessBoard, row, column):
    
   #non consente il posizionamento della regina se non vengono soddisfatti i vincoli 
    if checkColumn(chessBoard, row, column) and checkDiagonal(chessBoard, row, column):
        return True
    else:
        return False
    
if __name__ == '__main__':
 
    play = True
    
    print("Benvenuto, nel mondo delle Queens")
    
    while play:
        valid = False

        while not valid: 
            try:
                N = int(input("Perfavore inserisce la dimensione della board: "))
                valid = True
            except ValueError:
                print('Solo numeri') 
        
        print()
        
        # spazio vuoto nel tabelone
        board = [['-' for x in range(N)] for y in range(N)]
        
        separator = ''
        for i in range(len(board)):
            separator += '──'
            
        print(separator)
        
        #contatore per il numero totale di soluzioni
        counter = 0
        
        # funzione ricorsiva per trovare tutte le soluzioni
        # posizionamento Queen(board, 0)
        conteggioSol(board, 0)
        if N != 2 and N!= 3:
            placeQueen(board)
        
        #numero totale di soluzioni
        print("\nThere was a total of " + str(counter) + " solutions.")
        
        #ask vuoi giocare ancora?
        again = str(input("Vuoi giocare di nuovo? (S/N): "))
        if again.upper() == "N":
            play = False
