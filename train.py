import os
import pandas as pd
import numpy as np
import torch
from agent import DQNAgent

# Configuration
SPREAD_COST = 0.00015  # 1.5 pips transaction cost simulation
EPOCHS = 8
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 200

def load_and_preprocess_data(filepath):
    """Load MT5 CSV and compute indicator ratios."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")
        
    print(f"Loading data from {filepath}...")
    # MT5 export columns: Date, Open, High, Low, Close, Volume, Spread
    df = pd.read_csv(filepath, names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread'], encoding='utf-16')
    
    # Calculate SMA 200
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate ATR 14
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    df['TR'] = tr
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    
    # Drop rows with NaNs (first 200 rows due to SMA)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # State space inputs
    df['SMA_Ratio'] = (df['Close'] - df['SMA_200']) / df['Close']
    df['ATR_Ratio'] = df['ATR_14'] / df['Close']
    
    # Price difference to calculate rewards
    df['Price_Diff'] = df['Close'].shift(-1) - df['Close']
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

class TradingEnvSim:
    """Trading Simulation Environment for Offline RL Training."""
    def __init__(self, df):
        self.df = df
        self.reset()
        
    def reset(self):
        self.current_idx = 0
        self.position = 0  # 0 = Flat, 1 = Buy, -1 = Sell
        row = self.df.iloc[self.current_idx]
        self.state = np.array([row['SMA_Ratio'], row['ATR_Ratio'], float(self.position)], dtype=np.float32)
        return self.state
        
    def step(self, action):
        """Execute one step in the environment."""
        # Actions: 0 = Flat, 1 = Buy, 2 = Sell
        # Convert action to position type:
        # Action 0 -> Position 0 (Flat)
        # Action 1 -> Position 1 (Buy)
        # Action 2 -> Position -1 (Sell)
        target_position = 0
        if action == 1:
            target_position = 1
        elif action == 2:
            target_position = -1
            
        row = self.df.iloc[self.current_idx]
        price_diff = row['Price_Diff']
        
        # Calculate reward
        reward = 0.0
        # If position is BUY, we benefit from upward price changes
        if self.position == 1:
            reward += price_diff
        # If position is SELL, we benefit from downward price changes
        elif self.position == -1:
            reward -= price_diff
            
        # Deduct transaction cost if position changes
        position_changed = (self.position != target_position)
        if position_changed:
            reward -= SPREAD_COST
            self.position = target_position
            
        # Move to next bar
        self.current_idx += 1
        done = (self.current_idx >= len(self.df) - 1)
        
        # Next state
        if not done:
            next_row = self.df.iloc[self.current_idx]
            next_state = np.array([next_row['SMA_Ratio'], next_row['ATR_Ratio'], float(self.position)], dtype=np.float32)
        else:
            next_state = self.state
            
        return next_state, reward, done

def evaluate_agent(agent, df, label="Evaluation"):
    """Evaluate agent performance on a dataset."""
    env = TradingEnvSim(df)
    state = env.reset()
    done = False
    
    total_reward = 0.0
    trades_count = 0
    buy_count = 0
    sell_count = 0
    flat_count = 0
    
    last_pos = 0
    
    while not done:
        action = agent.act(state, eval_mode=True)
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        if last_pos != env.position:
            trades_count += 1
        
        if env.position == 1:
            buy_count += 1
        elif env.position == -1:
            sell_count += 1
        else:
            flat_count += 1
            
        last_pos = env.position
        state = next_state
        
    print(f"\n[{label} Results]")
    print(f"  Total Simulated Return: {total_reward:+.5f} (in price points)")
    print(f"  Total Position Shifts:  {trades_count}")
    print(f"  Market Exposure:        BUY={buy_count} bars, SELL={sell_count} bars, FLAT={flat_count} bars")
    return total_reward

def main():
    # Load Datasets
    train_path = r"C:\Users\nnadi\Documents\Work\mql5\EA\RL learning\training_data\EURUSDH1.csv"
    test_path = r"C:\Users\nnadi\Documents\Work\mql5\EA\RL learning\testing_data\EURUSDH1.csv"
    
    try:
        train_df = load_and_preprocess_data(train_path)
        test_df = load_and_preprocess_data(test_path)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
        
    # State dimension = 3 [SMA_Ratio, ATR_Ratio, Position_Status]
    # Action dimension = 3 [Flat, Buy, Sell]
    state_dim = 3
    action_dim = 3
    
    agent = DQNAgent(state_dim, action_dim)
    env = TradingEnvSim(train_df)
    
    print("\nStarting Offline PyTorch DQN Training...")
    best_test_reward = -9999.0
    
    for epoch in range(EPOCHS):
        state = env.reset()
        done = False
        epoch_reward = 0.0
        losses = []
        
        step_count = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.cache(state, action, reward, next_state, done)
            # Optimize training time by learning every 10 steps
            if step_count % 10 == 0:
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                
            epoch_reward += reward
            state = next_state
            step_count += 1
            
            if step_count % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
                
        # Epoch metrics
        mean_loss = np.mean(losses) if len(losses) > 0 else 0.0
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Return: {epoch_reward:+.5f} | Loss: {mean_loss:.8f} | Epsilon: {agent.epsilon:.4f}")
        
        # Evaluate on test set
        test_reward = evaluate_agent(agent, test_df, label=f"Epoch {epoch+1} Validation")
        
        # Save best model
        if test_reward > best_test_reward:
            best_test_reward = test_reward
            agent.save("model.pth")
            print("  --> Saved new best model to model.pth!")
            
    print("\nTraining completed successfully! Best test reward:", best_test_reward)

if __name__ == "__main__":
    main()
