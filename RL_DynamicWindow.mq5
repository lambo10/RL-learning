//+------------------------------------------------------------------+
//|                                              RL_DynamicWindow.mq5|
//|                                  Copyright 2026, Trading Agent   |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Trading Agent"
#property link      "https://www.mql5.com"
#property version   "1.00"

//--- Includes
#include <Trade\Trade.mqh>

//--- Input parameters - RL Configuration
input group "--- Q-Learning Settings ---"
input double   InpAlpha             = 0.1;       // Learning Rate (Alpha)
input double   InpGamma             = 0.9;       // Discount Factor (Gamma)
input double   InpEpsilon           = 0.2;       // Exploration Rate (Epsilon)
input double   InpTradePenalty      = 1.0;       // Trade Execution Penalty (Reward reduction)
input bool     InpLoadQTable        = true;      // Load Q-Table on startup
input bool     InpSaveQTable        = true;      // Save Q-Table on shutdown
input string   InpFileName          = "";        // Custom Q-Table Filename (blank for auto)

input group "--- Volatility State Thresholds ---"
input double   InpVolThresholdLow   = 0.0010;    // Low Volatility Threshold (ATR / Close)
input double   InpVolThresholdHigh  = 0.0025;    // High Volatility Threshold (ATR / Close)

input group "--- Trading Settings ---"
input double   InpLotSize           = 0.1;       // Trading Lot Size
input int      InpStopLoss          = 300;       // Stop Loss in Points (0 for none)
input int      InpTakeProfit        = 600;       // Take Profit in Points (0 for none)
input ulong    InpMagicNumber       = 123456;    // Expert Advisor Magic Number
input int      InpSlippage          = 3;         // Allowed Slippage in Points

//--- RL Definitions
#define NUM_TREND_STATES  2
#define NUM_VOL_STATES    3
#define NUM_WINDOW_STATES 5
#define NUM_ACTIONS       9

//--- Window Size Options (Days)
const int WindowDays[NUM_WINDOW_STATES] = {3, 5, 7, 10, 14};

//--- Action Mapping Enums
enum ENUM_TRADE_ACTION 
{ 
   ACTION_FLAT = 0, 
   ACTION_BUY  = 1, 
   ACTION_SELL = 2 
};

enum ENUM_WINDOW_ACTION 
{ 
   WINDOW_DECREASE = 0, 
   WINDOW_HOLD     = 1, 
   WINDOW_INCREASE = 2 
};

//--- Global Variables
double QTable[NUM_TREND_STATES][NUM_VOL_STATES][NUM_WINDOW_STATES][NUM_ACTIONS];
int    currentWindowIndex = 2; // Start with 7 days (index 2 of WindowDays)
datetime windowStartTime;
double   startingBalance;
double   windowPeakEquity;
double   windowMaxDrawdown;
int      tradesCountInWindow;
bool     firstActionTaken = false;

//--- State Tracker for policy updates
int lastTrendState;
int lastVolState;
int lastWindowState;
int lastAction;

//--- Indicator Handles
int smaHandle = INVALID_HANDLE;
int atrHandle = INVALID_HANDLE;

//--- Trade Object
CTrade trade;

//+------------------------------------------------------------------+
//| Helper to normalize prices to valid tick increments              |
//+------------------------------------------------------------------+
double NormalizePrice(double price)
{
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickSize == 0.0) return NormalizeDouble(price, _Digits);
   return NormalizeDouble(MathRound(price / tickSize) * tickSize, _Digits);
}

//+------------------------------------------------------------------+
//| Get EA's current active position type                            |
//+------------------------------------------------------------------+
ENUM_TRADE_ACTION GetCurrentPositionType()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == _Symbol)
      {
         ulong magic = PositionGetInteger(POSITION_MAGIC);
         if(magic == InpMagicNumber)
         {
            long type = PositionGetInteger(POSITION_TYPE);
            if(type == POSITION_TYPE_BUY) return ACTION_BUY;
            if(type == POSITION_TYPE_SELL) return ACTION_SELL;
         }
      }
   }
   return ACTION_FLAT;
}

//+------------------------------------------------------------------+
//| Close all positions belonging to this EA                         |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == _Symbol)
      {
         ulong magic = PositionGetInteger(POSITION_MAGIC);
         if(magic == InpMagicNumber)
         {
            ulong ticket = PositionGetInteger(POSITION_TICKET);
            if(!trade.PositionClose(ticket))
            {
               Print("Error closing position ticket ", ticket, ". Code: ", trade.ResultRetcode(), ", Description: ", trade.ResultRetcodeDescription());
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Get actual elapsed trading days between two times                |
//+------------------------------------------------------------------+
int GetElapsedTradingDays(datetime start, datetime end)
{
   int shiftStart = iBarShift(_Symbol, PERIOD_D1, start, false);
   int shiftEnd = iBarShift(_Symbol, PERIOD_D1, end, false);
   if(shiftStart < 0 || shiftEnd < 0) return 0;
   return MathMax(0, shiftStart - shiftEnd);
}

//+------------------------------------------------------------------+
//| Check if both indicator handles have valid data ready            |
//+------------------------------------------------------------------+
bool IsIndicatorsReady()
{
   double sma[];
   if(CopyBuffer(smaHandle, 0, 0, 1, sma) <= 0) return false;
   double atr[];
   if(CopyBuffer(atrHandle, 0, 0, 1, atr) <= 0) return false;
   return true;
}

//+------------------------------------------------------------------+
//| Get Trend State (0 = Price < SMA 200, 1 = Price >= SMA 200)      |
//+------------------------------------------------------------------+
int GetTrendState()
{
   double sma[];
   if(CopyBuffer(smaHandle, 0, 0, 1, sma) <= 0)
   {
      Print("Error copying SMA buffer: ", GetLastError());
      return 0;
   }
   
   double close[];
   if(CopyClose(_Symbol, _Period, 0, 1, close) <= 0)
   {
      Print("Error copying Close price: ", GetLastError());
      return 0;
   }
   
   return (close[0] >= sma[0]) ? 1 : 0;
}

//+------------------------------------------------------------------+
//| Get Volatility State (0 = Low, 1 = Medium, 2 = High)             |
//+------------------------------------------------------------------+
int GetVolatilityState()
{
   double atr[];
   if(CopyBuffer(atrHandle, 0, 0, 1, atr) <= 0)
   {
      Print("Error copying ATR buffer: ", GetLastError());
      return 0;
   }
   
   double close[];
   if(CopyClose(_Symbol, _Period, 0, 1, close) <= 0)
   {
      Print("Error copying Close price: ", GetLastError());
      return 0;
   }
   
   if(close[0] == 0.0) return 0;
   
   double ratio = atr[0] / close[0];
   if(ratio < InpVolThresholdLow)
      return 0;
   else if(ratio < InpVolThresholdHigh)
      return 1;
   else
      return 2;
}

//+------------------------------------------------------------------+
//| Resolve default or parameterized Q-Table file path               |
//+------------------------------------------------------------------+
string GetQTableFileName()
{
   if(InpFileName != "") return InpFileName;
   return "RL_DynamicWindow_QTable_" + _Symbol + "_" + EnumToString(_Period) + "_" + string(InpMagicNumber) + ".bin";
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize Q-Table to zero
   ArrayInitialize(QTable, 0.0);
   
   // Set magic number and slippage on trade object
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpSlippage);
   
   // Create SMA 200 handle
   smaHandle = iMA(_Symbol, _Period, 200, 0, MODE_SMA, PRICE_CLOSE);
   if(smaHandle == INVALID_HANDLE)
   {
      Print("Failed to create SMA 200 handle. Error: ", GetLastError());
      return(INIT_FAILED);
   }
   
   // Create ATR 14 handle
   atrHandle = iATR(_Symbol, _Period, 14);
   if(atrHandle == INVALID_HANDLE)
   {
      Print("Failed to create ATR 14 handle. Error: ", GetLastError());
      return(INIT_FAILED);
   }
   
   // Input Parameter Validation Warnings
   if(InpAlpha < 0.0 || InpAlpha > 1.0)
      Print("Warning: InpAlpha is out of recommended range [0.0, 1.0]. Value: ", InpAlpha);
   if(InpGamma < 0.0 || InpGamma > 1.0)
      Print("Warning: InpGamma is out of recommended range [0.0, 1.0]. Value: ", InpGamma);
   if(InpEpsilon < 0.0 || InpEpsilon > 1.0)
      Print("Warning: InpEpsilon is out of recommended range [0.0, 1.0]. Value: ", InpEpsilon);
      
   // Load QTable from file if enabled
   if(InpLoadQTable)
   {
      string fileName = GetQTableFileName();
      if(FileIsExist(fileName, FILE_COMMON))
      {
         int fileHandle = FileOpen(fileName, FILE_READ | FILE_BIN | FILE_COMMON);
         if(fileHandle != INVALID_HANDLE)
         {
            uint read = FileReadArray(fileHandle, QTable);
            FileClose(fileHandle);
            if(read > 0)
            {
               Print("Successfully loaded Q-Table from common files: ", fileName);
            }
            else
            {
               Print("Warning: Loaded Q-Table file, but data was empty or invalid.");
            }
         }
         else
         {
            Print("Failed to open Q-Table file for reading. Error: ", GetLastError());
         }
      }
      else
      {
         Print("No existing Q-Table file found. Initializing learning from scratch.");
      }
   }
   
   firstActionTaken = false;
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Release indicator handles
   if(smaHandle != INVALID_HANDLE) IndicatorRelease(smaHandle);
   if(atrHandle != INVALID_HANDLE) IndicatorRelease(atrHandle);
   
   // Save QTable to file if enabled
   if(InpSaveQTable)
   {
      string fileName = GetQTableFileName();
      int fileHandle = FileOpen(fileName, FILE_WRITE | FILE_BIN | FILE_COMMON);
      if(fileHandle != INVALID_HANDLE)
      {
         uint written = FileWriteArray(fileHandle, QTable);
         FileClose(fileHandle);
         if(written > 0)
         {
            Print("Successfully saved Q-Table to common files: ", fileName);
         }
         else
         {
            Print("Error: Failed to write Q-Table array data to file.");
         }
      }
      else
      {
         Print("Failed to open Q-Table file for writing. Error: ", GetLastError());
      }
   }
}

//+------------------------------------------------------------------+
//| Select Action using Epsilon-Greedy Strategy with tie breaking    |
//+------------------------------------------------------------------+
int SelectAction(int trendState, int volState, int windowState)
{
   // Exploration
   double randVal = (double)MathRand() / 32767.0;
   if(randVal < InpEpsilon)
   {
      int action = MathRand() % NUM_ACTIONS;
      Print("Exploration: Selected random action ", action);
      return action;
   }
   
   // Exploitation
   int bestAction = 0;
   double maxValue = QTable[trendState][volState][windowState][0];
   
   // Track tied actions for random tie-breaking
   int bestActions[NUM_ACTIONS];
   int count = 0;
   
   for(int i = 0; i < NUM_ACTIONS; i++)
   {
      double val = QTable[trendState][volState][windowState][i];
      if(val > maxValue)
      {
         maxValue = val;
         bestAction = i;
         count = 0;
         bestActions[count] = i;
         count++;
      }
      else if(val == maxValue)
      {
         bestActions[count] = i;
         count++;
      }
   }
   
   // If multiple actions have the same maximal Q-value, select randomly among them
   if(count > 1)
   {
      bestAction = bestActions[MathRand() % count];
      Print("Exploitation: Tie-breaker selected action ", bestAction, " (out of ", count, " equal options) with Q-value = ", maxValue);
   }
   else
   {
      Print("Exploitation: Selected best action ", bestAction, " with Q-value = ", maxValue);
   }
   
   return bestAction;
}

//+------------------------------------------------------------------+
//| Execute Trading component of selected action                     |
//+------------------------------------------------------------------+
void ExecuteTradeAction(int tradeAct)
{
   ENUM_TRADE_ACTION currentPosType = GetCurrentPositionType();
   
   if(tradeAct == ACTION_FLAT)
   {
      if(currentPosType != ACTION_FLAT)
      {
         Print("Action: FLAT. Closing existing position.");
         CloseAllPositions();
      }
   }
   else if(tradeAct == ACTION_BUY)
   {
      if(currentPosType == ACTION_SELL)
      {
         Print("Action: BUY. Closing opposite SELL position first.");
         CloseAllPositions();
         currentPosType = ACTION_FLAT;
      }
      
      if(currentPosType == ACTION_FLAT)
      {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl = (InpStopLoss > 0) ? NormalizePrice(ask - InpStopLoss * _Point) : 0.0;
         double tp = (InpTakeProfit > 0) ? NormalizePrice(ask + InpTakeProfit * _Point) : 0.0;
         
         Print("Action: BUY. Opening BUY position. Lot: ", InpLotSize, ", SL: ", sl, ", TP: ", tp);
         if(trade.Buy(InpLotSize, _Symbol, ask, sl, tp, "RL Buy"))
         {
            tradesCountInWindow++;
         }
         else
         {
            Print("Failed to execute BUY. Code: ", trade.ResultRetcode(), ", Description: ", trade.ResultRetcodeDescription());
         }
      }
      else
      {
         Print("Action: BUY. Position already open, holding.");
      }
   }
   else if(tradeAct == ACTION_SELL)
   {
      if(currentPosType == ACTION_BUY)
      {
         Print("Action: SELL. Closing opposite BUY position first.");
         CloseAllPositions();
         currentPosType = ACTION_FLAT;
      }
      
      if(currentPosType == ACTION_FLAT)
      {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = (InpStopLoss > 0) ? NormalizePrice(bid + InpStopLoss * _Point) : 0.0;
         double tp = (InpTakeProfit > 0) ? NormalizePrice(bid - InpTakeProfit * _Point) : 0.0;
         
         Print("Action: SELL. Opening SELL position. Lot: ", InpLotSize, ", SL: ", sl, ", TP: ", tp);
         if(trade.Sell(InpLotSize, _Symbol, bid, sl, tp, "RL Sell"))
         {
            tradesCountInWindow++;
         }
         else
         {
            Print("Failed to execute SELL. Code: ", trade.ResultRetcode(), ", Description: ", trade.ResultRetcodeDescription());
         }
      }
      else
      {
         Print("Action: SELL. Position already open, holding.");
      }
   }
}

//+------------------------------------------------------------------+
//| Execute joint action (Window resizing + trade action)            |
//+------------------------------------------------------------------+
void ExecuteAction(int actionIndex)
{
   int tradeAct = actionIndex / 3;
   int windowAct = actionIndex % 3;

   // 1. Execute Window Meta-Action
   int oldWindowDays = WindowDays[currentWindowIndex];
   if(windowAct == WINDOW_DECREASE)
   {
      currentWindowIndex = MathMax(0, currentWindowIndex - 1);
   }
   else if(windowAct == WINDOW_INCREASE)
   {
      currentWindowIndex = MathMin(NUM_WINDOW_STATES - 1, currentWindowIndex + 1);
   }
   
   Print("Window Meta-Action: Code ", windowAct, ". Resized window from ", oldWindowDays, " to ", WindowDays[currentWindowIndex], " trading days.");

   // 2. Execute Trading Action
   ExecuteTradeAction(tradeAct);
}

//+------------------------------------------------------------------+
//| Q-Table Bellman Update on Window Expiration                      |
//+------------------------------------------------------------------+
void UpdatePolicy()
{
   datetime currentTime = TimeCurrent();
   int currentWindowDays = WindowDays[currentWindowIndex];
   int elapsedDays = GetElapsedTradingDays(windowStartTime, currentTime);
   
   if(elapsedDays >= currentWindowDays)
   {
      Print("--- Lookback Window Expired. Elapsed trading days: ", elapsedDays, ". Running policy update. ---");
      
      // 1. Calculate Reward
      double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      double netProfit = currentBalance - startingBalance;
      
      double reward = 0.0;
      if(windowMaxDrawdown > 0.001)
      {
         if(netProfit >= 0.0)
         {
            // Profit ratio relative to drawdown
            reward = netProfit / windowMaxDrawdown;
         }
         else
         {
            // Penalize negative performance. Subtracting drawdown ensures larger drawdowns get punished more.
            reward = netProfit - windowMaxDrawdown;
         }
      }
      else
      {
         reward = netProfit;
      }
      
      // Deduct over-trading penalty
      double penalty = tradesCountInWindow * InpTradePenalty;
      reward -= penalty;
      
      Print("Reward Stats: Net Profit = ", DoubleToString(netProfit, 2), 
            ", Max Drawdown = ", DoubleToString(windowMaxDrawdown, 2), 
            ", Trades Executed = ", tradesCountInWindow, 
            ", Penalty = ", DoubleToString(penalty, 2), 
            ", Final Reward = ", DoubleToString(reward, 4));

      // 2. Observe the next state
      int nextTrendState = GetTrendState();
      int nextVolState = GetVolatilityState();
      int nextWindowState = currentWindowIndex;
      
      // 3. Find max Q-value for the next state
      double maxNextQ = QTable[nextTrendState][nextVolState][nextWindowState][0];
      for(int i = 1; i < NUM_ACTIONS; i++)
      {
         if(QTable[nextTrendState][nextVolState][nextWindowState][i] > maxNextQ)
         {
            maxNextQ = QTable[nextTrendState][nextVolState][nextWindowState][i];
         }
      }
      
      // 4. Bellman equation update for the state-action pair we started the window with
      double oldQ = QTable[lastTrendState][lastVolState][lastWindowState][lastAction];
      QTable[lastTrendState][lastVolState][lastWindowState][lastAction] += InpAlpha * (reward + InpGamma * maxNextQ - oldQ);
      double newQ = QTable[lastTrendState][lastVolState][lastWindowState][lastAction];
      
      Print("Bellman Update: QTable[Trend:", lastTrendState, "][Vol:", lastVolState, "][WindowIdx:", lastWindowState, "][Action:", lastAction, 
            "] updated from ", DoubleToString(oldQ, 4), " to ", DoubleToString(newQ, 4));

      // 5. Select next joint action
      int nextAction = SelectAction(nextTrendState, nextVolState, nextWindowState);
      
      // 6. Execute action (applies window adjustment and handles position opening/closing)
      ExecuteAction(nextAction);
      
      // 7. Store current states as 'last' for the next lookback window transition
      lastTrendState = nextTrendState;
      lastVolState = nextVolState;
      lastWindowState = currentWindowIndex; // Store actual window size state for upcoming period
      lastAction = nextAction;
      
      // 8. Reset window performance tracking variables
      windowStartTime = currentTime;
      startingBalance = currentBalance;
      windowPeakEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      windowMaxDrawdown = 0.0;
      tradesCountInWindow = 0;
      
      // 9. Save Q-Table to file to secure learning progress (highly useful during live trading or backtesting)
      if(InpSaveQTable)
      {
         string fileName = GetQTableFileName();
         int fileHandle = FileOpen(fileName, FILE_WRITE | FILE_BIN | FILE_COMMON);
         if(fileHandle != INVALID_HANDLE)
         {
            FileWriteArray(fileHandle, QTable);
            FileClose(fileHandle);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // 1. Check indicators readiness and execute the first action to launch the loop
   if(!firstActionTaken)
   {
      if(IsIndicatorsReady())
      {
         int trend = GetTrendState();
         int vol = GetVolatilityState();
         int windowState = currentWindowIndex;
         
         int firstAction = SelectAction(trend, vol, windowState);
         ExecuteAction(firstAction);
         
         lastTrendState = trend;
         lastVolState = vol;
         lastWindowState = windowState;
         lastAction = firstAction;
         
         windowStartTime = TimeCurrent();
         startingBalance = AccountInfoDouble(ACCOUNT_BALANCE);
         windowPeakEquity = AccountInfoDouble(ACCOUNT_EQUITY);
         windowMaxDrawdown = 0.0;
         tradesCountInWindow = 0;
         
         firstActionTaken = true;
         Print("RL Agent Initialized. Initial State: Trend = ", trend, ", Vol = ", vol, ", Window State = ", windowState, 
               " (", WindowDays[windowState], " days). Initial Action = ", firstAction);
      }
      else
      {
         // Skip tick until indicator data is populated
         return; 
      }
   }
   
   // 2. Continuous drawdown tracking during the window
   double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(currentEquity > windowPeakEquity)
   {
      windowPeakEquity = currentEquity;
   }
   double currentDrawdown = windowPeakEquity - currentEquity;
   if(currentDrawdown > windowMaxDrawdown)
   {
      windowMaxDrawdown = currentDrawdown;
   }
   
   // 3. Evaluate lookback window expiration and policy updates
   UpdatePolicy();
}
//+------------------------------------------------------------------+
