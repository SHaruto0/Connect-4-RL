import torch
import numpy as np

class Connect4:
    def __init__(self, mode=None, agent=None, seed=42):
        np.random.seed(seed=seed)
        self.mode = mode

        self.board = np.zeros((6,7), dtype=np.int8)
        self.game_on = False
        self.current_player = None
        self.player_name = {
            1: "Player",
            -1: "AI"
        }

        self.Q_model = agent
        
    def play(self):
        self.game_on = True
        self.board = np.zeros((6,7), dtype=np.int8)

        if self.Q_model is None:
            ValueError("Q Model Needed!!")

        print("Welcome to Connect 4!\n")
        while self.current_player is None:
            order = input("Going first or second (1 or 2): ")
            self.current_player = 1 if order == "1" else -1 if order == "2" else None

        while self.game_on:
            name = self.player_name[self.current_player]

            self.print_board()

            print(f"\n{name}'s turn:")
            
            if self.current_player == 1:
                column = None
                while column is None:
                    temp = int(input("Placed: "))
                    column = temp if 0 <= temp and temp <= 6 else None
            
                    row, placed = self.place(self.current_player, column)
                    column = column if placed else None
            
            if self.current_player == -1:
                with torch.no_grad():
                    obs = torch.tensor([self.get_board_state()], dtype=torch.float32, device=self.Q_model.device)
                    q_val = self.Q_model(obs).squeeze(0)  # shape (7,)

                    mask = torch.tensor(self.get_non_empty_column(), dtype=torch.bool, device=self.Q_model.device)
                    q_val[~mask] = -float('inf')

                    column = torch.argmax(q_val).item()
                row, placed = self.place(self.current_player, column)
                column = column if placed else None

            if column is not None:
                p, win = self.check_win(row, column)

                if win and p == 0:
                    print(f"TIE!")
                    self.game_on = False
                elif win:
                    print(f"{self.player_name[p]} won!")
                    self.game_on = False

                self.current_player *= -1
    
    def step(self, column, ai=True):
        reward = 0
        done = False
        
        if self.mode is None:
            print("Q Model Needed!!")
            raise ValueError("Q Model Needed!!")
        
        if self.current_player == 1:
            row, placed = self.place(self.current_player, column)
            column = column if placed else None

        elif self.current_player == -1 and self.mode == "random":
            column = np.random.choice(np.where(self.get_non_empty_column() == 1)[0].tolist())
            row, placed = self.place(self.current_player, column)
            column = column if placed else None

        elif self.current_player == -1 and self.mode == "selfplay":
            with torch.no_grad():
                obs = torch.tensor([self.get_board_state()], dtype=torch.float32, device=self.Q_model.device)
                q_val = self.Q_model(obs).squeeze(0)  # shape (7,)

                mask = torch.tensor(self.get_non_empty_column(), dtype=torch.bool, device=self.Q_model.device)
                q_val[~mask] = -float('inf')

                column = torch.argmax(q_val).item()

            row, placed = self.place(self.current_player, column)
            column = column if placed else None

        # self.print_board()
        if column is not None:
            p, win = self.check_win(row, column)

            if win and p == 1:
                reward = 1
                done = True
            elif win and p == -1:
                reward = -1
                done = True
            elif win and p == 0:
                reward = 0.5
                done = True

            self.current_player *= -1

            if not done and ai:
                board, reward, done = self.step(-1, False)
                return board, reward, done
            else:
                return self.get_board_state(), reward, done
        else:
            print(column)
            print("HIHIHI")

    def reset(self) -> list[int]:
        self.board = np.zeros((6,7), dtype=np.int8)
        state = self.get_board_state()

        self.current_player = np.random.choice([-1,1])

        if self.current_player == -1:
            self.step(-1, False)

        return state
    
    def get_board_state(self) -> list[int]:
        player1_flat = []
        player2_flat = []
        
        for row in self.board:
            for cell in row:
                player1_flat.append(1 if cell == 1 else 0)
                player2_flat.append(1 if cell == -1 else 0)
        
        return player1_flat + player2_flat

    def place(self, piece: int, cindex: int) -> bool:
        column = self.board[:,cindex]

        row = -1
        for i, c in enumerate(column):
            if c == 0:
                row = i
        if row == -1: return -1, False

        self.board[row, cindex] = piece
        
        return row, True
    
    def check_win(self, row: int, column: int) -> tuple[int, bool]:
        player = self.board[row][column]
        if player == 0:
            return 0, False

        directions = [
            (0, 1),  # horizontal
            (1, 0),  # vertical
            (1, 1),  # diagonal down-right
            (1, -1)  # diagonal down-left
        ]


        for dr, dc in directions:
            count = 1

            # Check in the positive direction
            r, c = row + dr, column + dc
            while 0 <= r < 6 and 0 <= c < 7 and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc

            # Check in the negative direction
            r, c = row - dr, column - dc
            while 0 <= r < 6 and 0 <= c < 7 and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 4:
                return self.current_player, True
        
        if np.sum(self.get_non_empty_column()) == 0:
            return 0, True
        return 0, False
    
    def get_non_empty_column(self) -> list[int]:
        return (self.board[0] == 0).astype(int)
    
    def print_board(self) -> None:
        print("\nCurrent board:")
        for row in self.board:
            print("|", end="")
            for cell in row:
                if cell == 0:
                    print("   |", end="")
                elif cell == 1:
                    print(" O |", end="")
                else:
                    print(" X |", end="")
            print()
        print("-" * 29)

        print(" ", end="")
        for i in range(7):
            print(f" {i} ", end=" ")
        print()


if __name__ == "__main__":
    game = Connect4(mode="random")

    game.play()