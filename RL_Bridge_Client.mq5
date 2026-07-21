//+------------------------------------------------------------------+
//|                                             RL_Bridge_Client.mq5|
//|                                  Copyright 2026, Trading Agent   |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Trading Agent"
#property link      "https://www.mql5.com"
#property version   "1.00"

//--- Includes
#include <Trade\Trade.mqh>

//--- Input parameters
input group "--- Python Server Configuration ---"
input string   InpServerUrl         = "http://127.0.0.1:8000/predict"; // Prediction Endpoint
input int      InpTimeoutMs         = 3000;                            // Timeout in Milliseconds

input group "--- Trading Settings ---"
input double   InpLotSize           = 0.1;                             // Trading Lot Size
input int      InpStopLoss          = 150;                             // Stop Loss in Points (0 for none)
input int      InpTakeProfit        = 300;                             // Take Profit in Points (0 for none)
input ulong    InpMagicNumber       = 654321;                          // Expert Magic Number
input int      InpSlippage          = 3;                               // Slippage in Points

//--- Global Variables
int      smaHandle = INVALID_HANDLE;
int      atrHandle = INVALID_HANDLE;
datetime lastBarTime = 0;

//--- Client Diagnostics
string   serverStatus = "Offline";
int      lastReceivedAction = -1;
string   lastTradeError = "";
int      networkRequestsCount = 0;
int      successfulRequestsCount = 0;

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
//| Get EA's current active position type (1 = Buy, -1 = Sell, 0=None)|
//+------------------------------------------------------------------+
double GetCurrentPositionStatus()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == _Symbol)
      {
         ulong magic = PositionGetInteger(POSITION_MAGIC);
         if(magic == InpMagicNumber)
         {
            long type = PositionGetInteger(POSITION_TYPE);
            if(type == POSITION_TYPE_BUY) return 1.0;
            if(type == POSITION_TYPE_SELL) return -1.0;
         }
      }
   }
   return 0.0;
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
               lastTradeError = StringFormat("Close Failed: %s", trade.ResultRetcodeDescription());
               Print("Error closing position. Code: ", trade.ResultRetcode());
            }
            else
            {
               lastTradeError = "";
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Send State to Python Server and Receive Action (0=Flat, 1=Buy, 2=Sell)|
//+------------------------------------------------------------------+
int QueryPythonModel(double close, double sma, double atr, double position, int &action)
{
   string headers = "Content-Type: application/json\r\n";
   
   // Formulate JSON body
   string body = StringFormat("{\"close\":%s,\"sma\":%s,\"atr\":%s,\"position\":%s}",
                              DoubleToString(close, 8),
                              DoubleToString(sma, 8),
                              DoubleToString(atr, 8),
                              DoubleToString(position, 1));
                              
   uchar data[];
   int len = StringToCharArray(body, data, 0, WHOLE_ARRAY, CP_UTF8);
   if(len > 1)
   {
      ArrayResize(data, len - 1); // Remove the null-terminating character '\0'
   }
   
   uchar result[];
   string responseHeaders = "";
   
   networkRequestsCount++;
   ResetLastError();
   
   // Send POST request
   int res = WebRequest("POST", InpServerUrl, headers, InpTimeoutMs, data, result, responseHeaders);
   
   if(res == -1)
   {
      int err = GetLastError();
      serverStatus = StringFormat("Error: GetLastError %d", err);
      return -1;
   }
   
   if(res == 200)
   {
      string responseJson = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
      
      // Parse "action" integer value from JSON: {"action": X, ...}
      int pos = StringFind(responseJson, "\"action\":");
      if(pos != -1)
      {
         int start = pos + 9;
         int end = start;
         while(end < StringLen(responseJson))
         {
            ushort c = StringGetCharacter(responseJson, end);
            if(c == ',' || c == '}' || c == ' ' || c == '\r' || c == '\n') break;
            end++;
         }
         string actionStr = StringSubstr(responseJson, start, end - start);
         action = (int)StringToInteger(actionStr);
         successfulRequestsCount++;
         serverStatus = "Connected (200 OK)";
         return 200;
      }
      else
      {
         serverStatus = "Error: JSON Action field missing";
         return -2;
      }
   }
   
   serverStatus = StringFormat("HTTP Error %d", res);
   return res;
}

//+------------------------------------------------------------------+
//| Execute action returned by Python model                          |
//+------------------------------------------------------------------+
void ExecuteModelAction(int action)
{
   double currentPos = GetCurrentPositionStatus();
   
   if(action == 0) // FLAT
   {
      if(currentPos != 0.0)
      {
         Print("Model Action: FLAT. Closing position.");
         CloseAllPositions();
      }
   }
   else if(action == 1) // BUY
   {
      if(currentPos == -1.0)
      {
         Print("Model Action: BUY. Closing opposite SELL position first.");
         CloseAllPositions();
         currentPos = 0.0;
      }
      
      if(currentPos == 0.0)
      {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl = (InpStopLoss > 0) ? NormalizePrice(ask - InpStopLoss * _Point) : 0.0;
         double tp = (InpTakeProfit > 0) ? NormalizePrice(ask + InpTakeProfit * _Point) : 0.0;
         
         PrintFormat("Model Action: BUY. Opening BUY trade. Lot: %.2f, SL: %.5f, TP: %.5f", InpLotSize, sl, tp);
         
         // Check execution mode (Market Execution does not allow SL/TP in opening request)
         long execMode = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_EXEMODE);
         if(execMode == SYMBOL_TRADE_EXECUTION_MARKET)
         {
            Print("ECN/Market Execution mode. Sending BUY without SL/TP first.");
            if(trade.Buy(InpLotSize, _Symbol, ask, 0.0, 0.0, "RL Bridge Buy"))
            {
               ulong code = trade.ResultRetcode();
               if(code == TRADE_RETCODE_DONE || code == TRADE_RETCODE_PLACED)
               {
                  lastTradeError = "";
                  Print("BUY deal executed. Adding SL/TP...");
                  Sleep(100); // Wait for position to register in terminal database
                  if(!trade.PositionModify(_Symbol, sl, tp))
                  {
                     Print("Warning: Could not modify position to set SL/TP. Code: ", trade.ResultRetcode(), " (", trade.ResultRetcodeDescription(), ")");
                  }
               }
               else
               {
                  lastTradeError = StringFormat("BUY Failed: %s", trade.ResultRetcodeDescription());
                  Print(lastTradeError);
               }
            }
            else
            {
               lastTradeError = StringFormat("BUY Rejected: %s", trade.ResultRetcodeDescription());
               Print(lastTradeError);
            }
         }
         else
         {
            // Instant / Request / Exchange execution (Stops allowed in order request)
            if(trade.Buy(InpLotSize, _Symbol, ask, sl, tp, "RL Bridge Buy"))
            {
               ulong code = trade.ResultRetcode();
               if(code == TRADE_RETCODE_DONE || code == TRADE_RETCODE_PLACED)
               {
                  lastTradeError = "";
               }
               else
               {
                  lastTradeError = StringFormat("BUY Failed: %s", trade.ResultRetcodeDescription());
                  Print(lastTradeError);
               }
            }
            else
            {
               lastTradeError = StringFormat("BUY Rejected: %s", trade.ResultRetcodeDescription());
               Print(lastTradeError);
            }
         }
      }
   }
   else if(action == 2) // SELL
   {
      if(currentPos == 1.0)
      {
         Print("Model Action: SELL. Closing opposite BUY position first.");
         CloseAllPositions();
         currentPos = 0.0;
      }
      
      if(currentPos == 0.0)
      {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl = (InpStopLoss > 0) ? NormalizePrice(bid + InpStopLoss * _Point) : 0.0;
         double tp = (InpTakeProfit > 0) ? NormalizePrice(bid - InpTakeProfit * _Point) : 0.0;
         
         PrintFormat("Model Action: SELL. Opening SELL trade. Lot: %.2f, SL: %.5f, TP: %.5f", InpLotSize, sl, tp);
         
         // Check execution mode
         long execMode = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_EXEMODE);
         if(execMode == SYMBOL_TRADE_EXECUTION_MARKET)
         {
            Print("ECN/Market Execution mode. Sending SELL without SL/TP first.");
            if(trade.Sell(InpLotSize, _Symbol, bid, 0.0, 0.0, "RL Bridge Sell"))
            {
               ulong code = trade.ResultRetcode();
               if(code == TRADE_RETCODE_DONE || code == TRADE_RETCODE_PLACED)
               {
                  lastTradeError = "";
                  Print("SELL deal executed. Adding SL/TP...");
                  Sleep(100);
                  if(!trade.PositionModify(_Symbol, sl, tp))
                  {
                     Print("Warning: Could not modify position to set SL/TP. Code: ", trade.ResultRetcode(), " (", trade.ResultRetcodeDescription(), ")");
                  }
               }
               else
               {
                  lastTradeError = StringFormat("SELL Failed: %s", trade.ResultRetcodeDescription());
                  Print(lastTradeError);
               }
            }
            else
            {
               lastTradeError = StringFormat("SELL Rejected: %s", trade.ResultRetcodeDescription());
               Print(lastTradeError);
            }
         }
         else
         {
            if(trade.Sell(InpLotSize, _Symbol, bid, sl, tp, "RL Bridge Sell"))
            {
               ulong code = trade.ResultRetcode();
               if(code == TRADE_RETCODE_DONE || code == TRADE_RETCODE_PLACED)
               {
                  lastTradeError = "";
               }
               else
               {
                  lastTradeError = StringFormat("SELL Failed: %s", trade.ResultRetcodeDescription());
                  Print(lastTradeError);
               }
            }
            else
            {
               lastTradeError = StringFormat("SELL Rejected: %s", trade.ResultRetcodeDescription());
               Print(lastTradeError);
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Update on-chart comment dashboard                                |
//+------------------------------------------------------------------+
void UpdateClientDashboard(double close, double sma, double atr)
{
   double smaRatio = (close - sma) / close;
   double atrRatio = atr / close;
   double posStatus = GetCurrentPositionStatus();
   
   string posName = "FLAT";
   if(posStatus == 1.0) posName = "BUY (Long)";
   else if(posStatus == -1.0) posName = "SELL (Short)";
   
   string actName = "NONE";
   if(lastReceivedAction == 0) actName = "FLAT";
   else if(lastReceivedAction == 1) actName = "BUY";
   else if(lastReceivedAction == 2) actName = "SELL";
   
   int nextRunSeconds = (int)(60 - (TimeCurrent() % 60));
   
   string text = StringFormat(
      "===========================================================\n"+
      "   PYTHON-MQL5 DRL BRIDGE CLIENT (1-MINUTE CYCLE)\n"+
      "===========================================================\n"+
      "Symbol: %s (M1-based)  |  Magic Number: %d\n"+
      "Server Connection Status: %s\n"+
      "-----------------------------------------------------------\n"+
      "MARKET METRICS (M1 Frame):\n"+
      "  - Last Close Price: %.5f\n"+
      "  - SMA 200:          %.5f (Ratio: %+7.5f)\n"+
      "  - ATR 14:           %.5f (Ratio: %7.5f)\n"+
      "-----------------------------------------------------------\n"+
      "AGENT CURRENT STATE & ACTIONS:\n"+
      "  - Current Position:  %s\n"+
      "  - Last Model Action: Action %d (%s)\n"+
      "  - Next Query In:     %d seconds\n"+
      "-----------------------------------------------------------\n"+
      "BRIDGE NETWORK STATISTICS:\n"+
      "  - Total Queries:     %d\n"+
      "  - Successful:        %d (Success Rate: %.1f%%)\n"+
      "%s"+
      "===========================================================",
      _Symbol, InpMagicNumber,
      serverStatus,
      close, sma, smaRatio, atr, atrRatio,
      posName,
      lastReceivedAction, actName,
      nextRunSeconds,
      networkRequestsCount,
      successfulRequestsCount,
      (networkRequestsCount > 0 ? ((double)successfulRequestsCount / networkRequestsCount) * 100.0 : 0.0),
      (lastTradeError != "" ? "-----------------------------------------------------------\n[WARNING] Last Order Error: " + lastTradeError + "\n" : "")
   );
   
   Comment(text);
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Set magic number and slippage
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpSlippage);
   
   // Create SMA 200 handle on M1 timeframe
   smaHandle = iMA(_Symbol, PERIOD_M1, 200, 0, MODE_SMA, PRICE_CLOSE);
   if(smaHandle == INVALID_HANDLE)
   {
      Print("Failed to create SMA handle on M1. Error: ", GetLastError());
      return(INIT_FAILED);
   }
   
   // Create ATR 14 handle on M1 timeframe
   atrHandle = iATR(_Symbol, PERIOD_M1, 14);
   if(atrHandle == INVALID_HANDLE)
   {
      Print("Failed to create ATR handle on M1. Error: ", GetLastError());
      return(INIT_FAILED);
   }
   
   lastBarTime = 0;
   serverStatus = "Initialized (Waiting for first bar)";
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Clear comment
   Comment("");
   
   // Release indicator handles
   if(smaHandle != INVALID_HANDLE) IndicatorRelease(smaHandle);
   if(atrHandle != INVALID_HANDLE) IndicatorRelease(atrHandle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // 1. Check if indicators have data ready
   double sma[];
   double atr[];
   double close[];
   
   if(CopyBuffer(smaHandle, 0, 1, 1, sma) <= 0 ||
      CopyBuffer(atrHandle, 0, 1, 1, atr) <= 0 ||
      CopyClose(_Symbol, PERIOD_M1, 1, 1, close) <= 0)
   {
      Comment("RL Client EA: Waiting for M1 indicator buffers to load...");
      return;
   }
   
   // 2. Check if a new M1 bar has opened
   datetime currentBarTime[];
   if(CopyTime(_Symbol, PERIOD_M1, 0, 1, currentBarTime) <= 0) return;
   
   if(currentBarTime[0] != lastBarTime)
   {
      // New M1 bar opened! query the server using the completed bar index 1
      double prevClose = close[0];
      double prevSma = sma[0];
      double prevAtr = atr[0];
      double posStatus = GetCurrentPositionStatus();
      
      int action = -1;
      int responseCode = QueryPythonModel(prevClose, prevSma, prevAtr, posStatus, action);
      
      if(responseCode == 200)
      {
         lastReceivedAction = action;
         
         string actName = "NONE";
         if(action == 0) actName = "FLAT";
         else if(action == 1) actName = "BUY";
         else if(action == 2) actName = "SELL";
         
         PrintFormat("Prediction Response Received -> Action: %d (%s) | Inputs: Close=%.5f, SMA=%.5f, ATR=%.5f, Position=%.1f", 
                     action, actName, prevClose, prevSma, prevAtr, posStatus);
                     
         ExecuteModelAction(action);
      }
      else
      {
         Print("Server Query failed. HTTP Code: ", responseCode, ", Status: ", serverStatus);
      }
      
      lastBarTime = currentBarTime[0];
   }
   
   // 3. Keep visual dashboard updated on every tick
   UpdateClientDashboard(close[0], sma[0], atr[0]);
}
//+------------------------------------------------------------------+
