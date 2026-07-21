import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np
import torch
from agent import DQNAgent

# Load the trained model
STATE_DIM = 3
ACTION_DIM = 3
MODEL_PATH = "model.pth"

agent = DQNAgent(STATE_DIM, ACTION_DIM)
if os.path.exists(MODEL_PATH):
    try:
        agent.load(MODEL_PATH)
        print(f"Loaded trained RL model from '{MODEL_PATH}'")
    except Exception as e:
        print(f"Error loading '{MODEL_PATH}': {e}. Using random policy.")
else:
    print(f"Warning: '{MODEL_PATH}' not found. Serving actions using random policy.")

class RLBridgeRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for serving MQL5 predictions."""
    def log_message(self, format, *args):
        # Suppress default server logs in terminal to keep outputs clean
        return
        
    def do_POST(self):
        if self.path == "/predict":
            # Read content length and parse JSON request
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                decoded_data = post_data.decode('utf-8').replace('\x00', '').strip()
                data = json.loads(decoded_data)
                
                # Extract features sent by MQL5
                close_val = float(data['close'])
                sma_val = float(data['sma'])
                atr_val = float(data['atr'])
                position = float(data['position'])  # 1 for Buy, -1 for Sell, 0 for Flat
                
                # Compute scale-invariant state features
                sma_ratio = (close_val - sma_val) / close_val if close_val != 0 else 0.0
                atr_ratio = atr_val / close_val if close_val != 0 else 0.0
                
                # Assemble state vector: [SMA_Ratio, ATR_Ratio, Position_Status]
                state = np.array([sma_ratio, atr_ratio, position], dtype=np.float32)
                
                # Query PyTorch DQN Agent for optimal action
                action = agent.act(state, eval_mode=True)
                
                # Map action back to friendly description
                action_name = "FLAT"
                if action == 1:
                    action_name = "BUY"
                elif action == 2:
                    action_name = "SELL"
                
                # Prepare JSON response
                response_data = {
                    "action": action,
                    "action_name": action_name,
                    "sma_ratio": sma_ratio,
                    "atr_ratio": atr_ratio
                }
                
                # Send HTTP 200 OK
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                
                print("\n" + "="*50)
                print("   INCOMING PREDICTION REQUEST")
                print("="*50)
                print(f"Raw Request Payload:   {decoded_data}")
                print(f"Computed State Ratios: SMA_Ratio={sma_ratio:+.6f}, ATR_Ratio={atr_ratio:.6f}, Position={position}")
                print(f"DQN Model Decision:    Action {action} ({action_name})")
                print(f"Sent Response JSON:    {json.dumps(response_data)}")
                print("="*50 + "\n")
                
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
                print(f"Error handling prediction request: {e}")
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8000):
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, RLBridgeRequestHandler)
    print(f"\nRL Python HTTP Bridge Server running on http://127.0.0.1:{port} ...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping HTTP Server...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
