import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime
import json
import os

class TradingRoutineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gold ML Trading Routine Assistant")
        self.root.geometry("800x600")
        self.root.configure(bg='#1e1e1e')
        
        # State tracking
        self.current_step = 0
        self.session_log = []
        self.trade_count = 0
        self.daily_loss = 0
        self.consecutive_losses = 0
        
        # Routine steps
        self.routine = [
            {
                "phase": "PRE-MARKET PREPARATION",
                "time": "7:00 AM - 8:00 AM ET",
                "question": "Have you checked the economic calendar for high-impact news today?",
                "yes_next": 1,
                "no_action": "stop",
                "no_message": "⛔ STOP: Check ForexFactory or Investing.com for news first."
            },
            {
                "phase": "PRE-MARKET PREPARATION",
                "time": "7:00 AM - 8:00 AM ET",
                "question": "Is there any major news event scheduled today (NFP, FOMC, CPI)?",
                "yes_next": "stop",
                "yes_message": "⛔ STOP: Do not trade on major news days. Close the system.",
                "no_next": 2
            },
            {
                "phase": "PRE-MARKET PREPARATION",
                "time": "7:00 AM - 8:00 AM ET",
                "question": "Have you run the ML model with yesterday's data?",
                "yes_next": 3,
                "no_action": "stop",
                "no_message": "⛔ STOP: Run the ML model first to get today's signal."
            },
            {
                "phase": "PRE-MARKET PREPARATION",
                "time": "7:00 AM - 8:00 AM ET",
                "question": "Did the model give a BUY signal?",
                "yes_next": 4,
                "no_action": "stop",
                "no_message": "⛔ STOP: Model says HOLD. No trading today. Close the system."
            },
            {
                "phase": "PRE-MARKET PREPARATION",
                "time": "7:00 AM - 8:00 AM ET",
                "question": "✅ BUY signal confirmed. Ready to proceed to market phase?",
                "yes_next": 5,
                "no_action": "stop",
                "no_message": "System paused. Resume when ready."
            },
            {
                "phase": "MARKET OPEN - WAIT PERIOD",
                "time": "8:00 AM - 9:00 AM ET",
                "question": "Is it currently between 8:00 AM - 9:00 AM ET (market open hour)?",
                "yes_next": 6,
                "no_next": 7
            },
            {
                "phase": "MARKET OPEN - WAIT PERIOD",
                "time": "8:00 AM - 9:00 AM ET",
                "question": "⏳ WAIT: Do not trade during first hour. Are you waiting?",
                "yes_next": 5,
                "no_action": "warning",
                "no_message": "⚠️ WARNING: Trading during first hour violates rules. Wait until 9:00 AM."
            },
            {
                "phase": "ACTIVE TRADING WINDOW",
                "time": "9:00 AM - 4:30 PM ET",
                "question": "Have you opened 4H gold chart and identified support/resistance levels?",
                "yes_next": 8,
                "no_action": "stop",
                "no_message": "⛔ STOP: Identify key levels before proceeding."
            },
            {
                "phase": "ACTIVE TRADING WINDOW",
                "time": "9:00 AM - 4:30 PM ET",
                "question": "Do you see a valid 4H entry condition? (pullback to support / breakout / MA bounce / RSI reversal)",
                "yes_next": 9,
                "no_next": 15,
                "no_message": "⏳ No setup yet. Keep monitoring. Click 'Check Again' when setup appears."
            },
            {
                "phase": "ACTIVE TRADING WINDOW",
                "time": "9:00 AM - 4:30 PM ET",
                "question": "Have you calculated position size: (Account × 2%) / stop-loss distance?",
                "yes_next": 10,
                "no_action": "stop",
                "no_message": "⛔ STOP: Calculate position size before entering trade."
            },
            {
                "phase": "ACTIVE TRADING WINDOW",
                "time": "9:00 AM - 4:30 PM ET",
                "question": "Have you set stop-loss below 4H support and take-profit at 1:3 RR minimum?",
                "yes_next": 11,
                "no_action": "stop",
                "no_message": "⛔ STOP: Set stop-loss and take-profit before entry."
            },
            {
                "phase": "ACTIVE TRADING WINDOW",
                "time": "9:00 AM - 4:30 PM ET",
                "question": "✅ Enter LONG position now. Trade entered?",
                "yes_next": 12,
                "no_action": "stop",
                "no_message": "Trade cancelled. Return to monitoring."
            },
            {
                "phase": "POSITION MANAGEMENT",
                "time": "While trade is active",
                "question": "Have you logged: entry price, stop-loss, take-profit, time, and reasoning?",
                "yes_next": 13,
                "no_action": "warning",
                "no_message": "⚠️ WARNING: Log trade details now for record keeping."
            },
            {
                "phase": "POSITION MANAGEMENT",
                "time": "While trade is active",
                "question": "Have you set price alerts at stop-loss and take-profit levels?",
                "yes_next": 14,
                "no_action": "warning",
                "no_message": "⚠️ Set alerts now. Do not watch screen constantly."
            },
            {
                "phase": "POSITION MANAGEMENT",
                "time": "While trade is active",
                "question": "✅ Trade active. Walk away. Has the trade closed (TP hit / SL hit / 4:30 PM reached)?",
                "yes_next": 16,
                "no_next": 14,
                "no_message": "Trade still active. Monitor alerts. Click 'Check Again' when trade closes."
            },
            {
                "phase": "MONITORING",
                "time": "9:00 AM - 4:30 PM ET",
                "question": "⏳ Continue monitoring 4H chart. Do you see a new valid entry setup?",
                "yes_next": 8,
                "no_next": 15,
                "no_message": "Keep monitoring. Return here when setup appears or day ends."
            },
            {
                "phase": "TRADE CLOSED",
                "time": "After trade exit",
                "question": "Did the trade result in a WIN?",
                "yes_next": 17,
                "no_next": 18
            },
            {
                "phase": "TRADE CLOSED",
                "time": "After trade exit",
                "question": "✅ Congratulations! Trade logged as WIN. Do you want to take a second trade today?",
                "yes_next": 19,
                "no_next": 20,
                "no_message": "Good decision. Done trading for today."
            },
            {
                "phase": "TRADE CLOSED",
                "time": "After trade exit",
                "question": "❌ Trade logged as LOSS. Have you logged what went wrong?",
                "yes_next": 21,
                "no_action": "warning",
                "no_message": "⚠️ Log the loss reason now for learning."
            },
            {
                "phase": "SECOND TRADE CHECK",
                "time": "After first trade",
                "question": "Is your total risk for the day still under 4% (2% per trade)?",
                "yes_next": 22,
                "no_action": "stop",
                "no_message": "⛔ STOP: Maximum daily risk reached. Done for today."
            },
            {
                "phase": "END OF DAY",
                "time": "5:00 PM ET",
                "question": "✅ Trading session complete. Ready to do end-of-day routine?",
                "yes_next": 23,
                "no_action": "stop",
                "no_message": "Complete end-of-day routine before closing."
            },
            {
                "phase": "LOSS CHECK",
                "time": "After loss",
                "question": "Was this your 3rd consecutive loss today?",
                "yes_next": "stop",
                "yes_message": "⛔ STOP TRADING: 3 consecutive losses. Close system. Return tomorrow fresh.",
                "no_next": 24
            },
            {
                "phase": "SECOND TRADE CHECK",
                "time": "Before second trade",
                "question": "Have you already taken 1 trade today?",
                "yes_next": 25,
                "no_next": 8
            },
            {
                "phase": "END OF DAY",
                "time": "5:00 PM ET",
                "question": "Have you closed all open positions at market price?",
                "yes_next": 26,
                "no_action": "stop",
                "no_message": "⛔ CRITICAL: Close all positions NOW. Rule: No overnight holds."
            },
            {
                "phase": "LOSS CHECK",
                "time": "After loss",
                "question": "Is your account down 5% or more today?",
                "yes_next": "stop",
                "yes_message": "⛔ STOP TRADING: Daily loss limit hit. Close system. Return tomorrow.",
                "no_next": 27
            },
            {
                "phase": "SECOND TRADE CHECK",
                "time": "Before second trade",
                "question": "Is the first trade already closed (not still open)?",
                "yes_next": 8,
                "no_action": "stop",
                "no_message": "⛔ STOP: Cannot take second trade while first is still open."
            },
            {
                "phase": "END OF DAY",
                "time": "5:00 PM ET",
                "question": "Have you logged all trades in your spreadsheet with full details?",
                "yes_next": 28,
                "no_action": "warning",
                "no_message": "⚠️ Log trades now before forgetting details."
            },
            {
                "phase": "LOSS CHECK",
                "time": "After loss",
                "question": "Do you feel emotional, angry, or want revenge?",
                "yes_next": "stop",
                "yes_message": "⛔ STOP TRADING: Emotional state detected. Walk away. Return tomorrow calm.",
                "no_next": 20
            },
            {
                "phase": "END OF DAY",
                "time": "5:00 PM ET",
                "question": "Have you calculated today's P&L percentage?",
                "yes_next": 29,
                "no_action": "warning",
                "no_message": "Calculate P&L now for tracking."
            },
            {
                "phase": "END OF DAY",
                "time": "5:00 PM ET",
                "question": "Have you reviewed if you followed ALL rules today?",
                "yes_next": "complete",
                "yes_message": "✅ SESSION COMPLETE! Well done. See you tomorrow.",
                "no_action": "warning",
                "no_message": "Review rule violations. Learn from them for tomorrow."
            }
        ]
        
        # UI Setup
        self.setup_ui()
        self.show_step()
    
    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg='#1e1e1e')
        title_frame.pack(pady=20)
        
        title_label = tk.Label(
            title_frame,
            text="🥇 GOLD ML TRADING ROUTINE ASSISTANT",
            font=("Arial", 20, "bold"),
            bg='#1e1e1e',
            fg='#FFD700'
        )
        title_label.pack()
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.root,
            length=700,
            mode='determinate'
        )
        self.progress.pack(pady=10)
        
        # Main content frame
        content_frame = tk.Frame(self.root, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        content_frame.pack(pady=20, padx=40, fill=tk.BOTH, expand=True)
        
        # Phase label
        self.phase_label = tk.Label(
            content_frame,
            text="",
            font=("Arial", 14, "bold"),
            bg='#2d2d2d',
            fg='#4CAF50',
            wraplength=700
        )
        self.phase_label.pack(pady=10)
        
        # Time label
        self.time_label = tk.Label(
            content_frame,
            text="",
            font=("Arial", 11),
            bg='#2d2d2d',
            fg='#9E9E9E'
        )
        self.time_label.pack()
        
        # Question label
        self.question_label = tk.Label(
            content_frame,
            text="",
            font=("Arial", 13),
            bg='#2d2d2d',
            fg='#FFFFFF',
            wraplength=700,
            justify=tk.LEFT
        )
        self.question_label.pack(pady=30)
        
        # Button frame
        button_frame = tk.Frame(content_frame, bg='#2d2d2d')
        button_frame.pack(pady=20)
        
        self.yes_button = tk.Button(
            button_frame,
            text="✅ YES",
            font=("Arial", 12, "bold"),
            bg='#4CAF50',
            fg='white',
            width=15,
            height=2,
            command=self.handle_yes
        )
        self.yes_button.pack(side=tk.LEFT, padx=10)
        
        self.no_button = tk.Button(
            button_frame,
            text="❌ NO",
            font=("Arial", 12, "bold"),
            bg='#F44336',
            fg='white',
            width=15,
            height=2,
            command=self.handle_no
        )
        self.no_button.pack(side=tk.LEFT, padx=10)
        
        # Status bar
        self.status_label = tk.Label(
            self.root,
            text=f"Step 1 of {len(self.routine)} | Trades: 0 | Time: {datetime.now().strftime('%I:%M %p')}",
            font=("Arial", 10),
            bg='#1e1e1e',
            fg='#9E9E9E'
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    
    def show_step(self):
        if self.current_step >= len(self.routine):
            self.complete_session()
            return
        
        step = self.routine[self.current_step]
        
        # Update UI
        self.phase_label.config(text=f"📍 {step['phase']}")
        self.time_label.config(text=f"⏰ {step['time']}")
        self.question_label.config(text=step['question'])
        
        # Update progress
        progress_value = (self.current_step / len(self.routine)) * 100
        self.progress['value'] = progress_value
        
        # Update status
        self.status_label.config(
            text=f"Step {self.current_step + 1} of {len(self.routine)} | Trades: {self.trade_count} | Time: {datetime.now().strftime('%I:%M %p')}"
        )
        
        # Log
        self.session_log.append({
            "step": self.current_step + 1,
            "phase": step['phase'],
            "question": step['question'],
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    def handle_yes(self):
        step = self.routine[self.current_step]
        
        # Check if YES leads to stop
        if step.get('yes_next') == 'stop':
            messagebox.showwarning("STOP", step.get('yes_message', 'Session ended.'))
            self.save_log()
            self.root.quit()
            return
        
        # Check if YES leads to completion
        if step.get('yes_next') == 'complete':
            messagebox.showinfo("SUCCESS", step.get('yes_message', 'Session complete!'))
            self.complete_session()
            return
        
        # Track trade count
        if 'Trade entered?' in step['question']:
            self.trade_count += 1
        
        # Move to next step
        next_step = step.get('yes_next', self.current_step + 1)
        self.current_step = next_step
        self.show_step()
    
    def handle_no(self):
        step = self.routine[self.current_step]
        
        # Check NO action
        action = step.get('no_action', 'next')
        message = step.get('no_message', '')
        
        if action == 'stop':
            messagebox.showerror("STOP", message)
            self.save_log()
            self.root.quit()
            return
        
        if action == 'warning':
            messagebox.showwarning("WARNING", message)
            # Stay on same step
            return
        
        # Check if NO has specific next step
        if 'no_next' in step:
            self.current_step = step['no_next']
        else:
            self.current_step += 1
        
        self.show_step()
    
    def complete_session(self):
        self.save_log()
        messagebox.showinfo(
            "SESSION COMPLETE",
            f"✅ Trading routine completed!\n\n"
            f"Trades taken: {self.trade_count}\n"
            f"Time: {datetime.now().strftime('%I:%M %p')}\n\n"
            f"Log saved. See you tomorrow!"
        )
        self.root.quit()
    
    def save_log(self):
        log_file = "trading_routine_log.json"
        session_data = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "start_time": self.session_log[0]['time'] if self.session_log else "N/A",
            "end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "trades_taken": self.trade_count,
            "steps_completed": len(self.session_log),
            "log": self.session_log
        }
        
        # Load existing logs
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(session_data)
        
        # Save
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"Session log saved to {log_file}")


if __name__ == "__main__":
    root = tk.Tk()
    app = TradingRoutineApp(root)
    root.mainloop()