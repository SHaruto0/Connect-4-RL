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

        self.rewards = {
            "continue": 0,
            "illegal": -1,
            "win": 1,
            "tie": 0.5,
            "loss": -1,
            # "reach": 0.2,
            # "block": 0.2
        }

        self.Q_model = agent
        self.turns = 0
        
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
            
            if self.current_player == -1 and self.Q_model is not None:
                with torch.no_grad():
                    obs = torch.tensor([self.get_board_state(-1)], dtype=torch.float32, device=self.Q_model.device)
                    q_val = self.Q_model(obs).squeeze(0)

                    mask = torch.tensor(self.get_non_empty_column(), dtype=torch.bool, device=self.Q_model.device)
                    q_val[~mask] = -float('inf')

                    column = torch.argmax(q_val).item()
                row, placed = self.place(self.current_player, column)
                column = column if placed else None
            elif self.current_player == -1:
                column = None
                while column is None:
                    temp = int(input("Placed: "))
                    column = temp if 0 <= temp and temp <= 6 else None
            
                    row, placed = self.place(self.current_player, column)
                    column = column if placed else None

            if column is not None:
                result, done = self.check_win(row, column)

                if done and result == "tie":
                    print(f"TIE!")
                    self.game_on = False
                elif done:
                    print(f"{self.player_name[self.current_player]} won!")
                    self.game_on = False

                self.current_player *= -1
        
        self.print_board()
    
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
                obs = torch.tensor([self.get_board_state(-1)], dtype=torch.float32, device=self.Q_model.device)
                q_val = self.Q_model(obs).squeeze(0)  # shape (7,)

                mask = torch.tensor(self.get_non_empty_column(), dtype=torch.bool, device=self.Q_model.device)
                q_val[~mask] = -float('inf')

                column = torch.argmax(q_val).item()

            row, placed = self.place(self.current_player, column)
            column = column if placed else None

        if column is None:
            print("Invalid move attempted.")
            return self.get_board_state(), self.rewards["illegal"], True

        # self.print_board()
        result, done = self.check_win(row, column)
        reward = self.rewards[result]

        # if not done:
        #     result, is_reach, reach_count = self.check_reach(row, column)
        #     reward += self.rewards[result] * reach_count
        #     # print(f"Player: {self.current_player} \t {is_reach = }")

        #     result, is_block, block_count = self.check_block(row, column)
        #     reward += self.rewards[result] * block_count
        #     # print(f"Player: {self.current_player} \t {is_block = }")

        self.current_player *= -1

        if not done and ai:
            board, ai_reward, done = self.step(-1, False)
            # reward -= ai_reward
            # if ai_reward == 1:
            #     reward = -1
            return board, reward, done
        else:
            return self.get_board_state(1), reward, done

    def reset(self) -> list[int]:
        self.board = np.zeros((6,7), dtype=np.int8)

        self.current_player = np.random.choice([-1,1])

        if self.current_player == -1:
            self.step(-1, False)

        return self.get_board_state(1)
    
    def get_board_state(self, player=None) -> list[int]:
        if player is None:
            player = self.current_player

        player_flat = []
        opponent_flat = []

        for row in self.board:
            for cell in row:
                if cell == player:
                    player_flat.append(1)
                    opponent_flat.append(0)
                elif cell == -player:
                    player_flat.append(0)
                    opponent_flat.append(1)
                else:
                    player_flat.append(0)
                    opponent_flat.append(0)

        return player_flat + opponent_flat

    def place(self, piece: int, cindex: int) -> bool:
        column = self.board[:,cindex]

        row = -1
        for i, c in enumerate(column):
            if c == 0:
                row = i
        if row == -1: return -1, False

        self.board[row, cindex] = piece
        
        return row, True
    
    def check_win(self, row: int, column: int) -> tuple[str, bool]:
        player = self.board[row][column]
        if player == 0:
            return "illegal", False

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

            if count >= 4 and self.current_player == 1:
                return "win", True
            elif count >= 4 and self.current_player == -1:
                return "loss", True
        
        if np.sum(self.get_non_empty_column()) == 0:
            return "tie", True
        return "continue", False

    def check_reach(self, row: int, column: int) -> tuple[str, bool]:
        player = self.board[row][column]
        if player == 0:
            return "illegal", False

        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal down-right
            (1, -1)   # diagonal down-left
        ]

        reach_count = 0
        for dr, dc in directions:
            line = []
            for i in range(-3, 4):
                r, c = row + dr * i, column + dc * i
                if 0 <= r < 6 and 0 <= c < 7:
                    line.append(self.board[r][c])
                else:
                    line.append(None)

            # Check every window of 4
            for i in range(len(line) - 3):
                window = line[i:i + 4]
                if None in window:
                    continue
                if window.count(player) == 3 and window.count(0) == 1:
                    reach_count += 1

        if reach_count != 0:
            return "reach", True, reach_count
        return "continue", False, 0


    def check_block(self, row: int, column: int) -> tuple[str, bool]:
        player = self.board[row][column]
        if player == 0:
            return "illegal", False

        opponent = -player
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal down-right
            (1, -1)   # diagonal down-left
        ]

        block_count = 0
        for dr, dc in directions:
            line = []
            coords = []  # store coordinates for each cell in the line
            for i in range(-3, 4):  # check up to 3 cells away in both directions
                r, c = row + dr * i, column + dc * i
                if 0 <= r < 6 and 0 <= c < 7:
                    line.append(self.board[r][c])
                    coords.append((r, c))
                else:
                    line.append(None)
                    coords.append(None)

            # Slide a window of 4 cells along that line
            for i in range(len(line) - 3):
                window = line[i:i + 4]
                positions = coords[i:i + 4]
                if None in window:
                    continue

                if window.count(opponent) == 3 and window.count(player) == 1:
                    empty_index = window.index(player)
                    if positions[empty_index] == (row, column):
                        block_count += 1

        if block_count != 0:
            return "block", True, block_count
        return "continue", False, 0


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