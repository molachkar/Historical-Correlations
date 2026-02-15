import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class TrainingAnalyzer:
    """Analyze TFT training results from log file"""
    
    def __init__(self, log_path="training.log"):
        self.log_path = log_path
        self.training_data = None
        self.config = {}
        self.results = {}
        
    def parse_log(self):
        """Extract training data from log file"""
        print("Parsing training log...")
        
        with open(self.log_path, 'r', encoding='utf-16') as f:
            content = f.read()
        
        # Extract configuration
        config_patterns = {
            'encoder_length': r'Encoder=(\d+)',
            'prediction_length': r'Predict=(\d+)',
            'hidden_size': r'Hidden=(\d+)',
            'device': r'Device:\s*(\w+)',
            'total_params': r'Params:\s*([\d,]+)',
            'total_rows': r'Loaded\s+(\d+)\s+rows',
            'total_features': r'Total features:\s*(\d+)',
            'train_samples': r'Train:\s*(\d+)\s*samples',
            'valid_samples': r'Valid:\s*(\d+)\s*samples',
            'target_mean': r'mean:\s*([\d.]+)%',
            'target_std': r'std:\s*([\d.]+)%',
        }
        
        for key, pattern in config_patterns.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(1).replace(',', '')
                try:
                    self.config[key] = float(value) if '.' in value else int(value)
                except:
                    self.config[key] = value
        
        # Extract epoch data
        epoch_pattern = r'Epoch\s+(\d+)/\d+\s+-\s+Train loss:\s+([\d.]+),\s+Val loss:\s+([\d.]+)'
        epochs = re.findall(epoch_pattern, content)
        
        if epochs:
            self.training_data = pd.DataFrame(epochs, columns=['epoch', 'train_loss', 'val_loss'])
            self.training_data = self.training_data.astype({
                'epoch': int,
                'train_loss': float,
                'val_loss': float
            })
        
        # Extract best model info
        best_match = re.search(r'Best val loss:\s*([\d.]+)', content)
        if best_match:
            self.results['best_val_loss'] = float(best_match.group(1))
            
        saved_models = re.findall(r'Saved best model:\s*(.+\.pth)', content)
        self.results['saved_models'] = saved_models
        
        early_stop = re.search(r'Early stopping at epoch (\d+)', content)
        self.results['early_stopped'] = bool(early_stop)
        self.results['early_stop_epoch'] = int(early_stop.group(1)) if early_stop else None
        
        print(f"✓ Parsed {len(self.training_data)} epochs")
        
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        
        print("\n📊 MODEL CONFIGURATION:")
        print(f"  • Encoder Length:      {self.config.get('encoder_length', 'N/A')} days")
        print(f"  • Prediction Length:   {self.config.get('prediction_length', 'N/A')} days")
        print(f"  • Hidden Size:         {self.config.get('hidden_size', 'N/A')}")
        print(f"  • Total Parameters:    {self.config.get('total_params', 'N/A'):,}")
        print(f"  • Device:              {self.config.get('device', 'N/A').upper()}")
        
        print("\n📈 DATASET INFO:")
        print(f"  • Total Rows:          {self.config.get('total_rows', 'N/A'):,}")
        print(f"  • Total Features:      {self.config.get('total_features', 'N/A')}")
        print(f"  • Training Samples:    {self.config.get('train_samples', 'N/A'):,}")
        print(f"  • Validation Samples:  {self.config.get('valid_samples', 'N/A'):,}")
        print(f"  • Target Mean:         {self.config.get('target_mean', 'N/A')}%")
        print(f"  • Target Std Dev:      {self.config.get('target_std', 'N/A')}%")
        
        print("\n🎯 TRAINING RESULTS:")
        print(f"  • Total Epochs Trained: {len(self.training_data)}")
        print(f"  • Early Stopping:       {'Yes (Epoch ' + str(self.results['early_stop_epoch']) + ')' if self.results['early_stopped'] else 'No'}")
        print(f"  • Best Val Loss:        {self.results.get('best_val_loss', 'N/A'):.4f}")
        
        if self.training_data is not None and len(self.training_data) > 0:
            final_train = self.training_data.iloc[-1]['train_loss']
            final_val = self.training_data.iloc[-1]['val_loss']
            print(f"  • Final Train Loss:     {final_train:.4f}")
            print(f"  • Final Val Loss:       {final_val:.4f}")
            
            # Loss improvement
            initial_train = self.training_data.iloc[0]['train_loss']
            initial_val = self.training_data.iloc[0]['val_loss']
            train_improvement = ((initial_train - final_train) / initial_train) * 100
            val_improvement = ((initial_val - self.results['best_val_loss']) / initial_val) * 100
            print(f"  • Train Loss Reduction: {train_improvement:.1f}%")
            print(f"  • Val Loss Reduction:   {val_improvement:.1f}%")
        
        print("\n💾 SAVED MODELS:")
        if self.results.get('saved_models'):
            for i, model in enumerate(self.results['saved_models'], 1):
                print(f"  {i}. {model}")
        else:
            print("  No models found in log")
        
        print("\n" + "="*70)
    
    def plot_training_curves(self, save_path='training_curves.png'):
        """Plot training and validation loss curves"""
        if self.training_data is None or len(self.training_data) == 0:
            print("No training data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Training vs Validation Loss
        ax1 = axes[0, 0]
        ax1.plot(self.training_data['epoch'], self.training_data['train_loss'], 
                'b-o', label='Training Loss', linewidth=2, markersize=6)
        ax1.plot(self.training_data['epoch'], self.training_data['val_loss'], 
                'r-s', label='Validation Loss', linewidth=2, markersize=6)
        
        # Mark best epoch
        best_epoch = self.training_data.loc[self.training_data['val_loss'].idxmin(), 'epoch']
        best_val = self.training_data['val_loss'].min()
        ax1.axvline(best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (Epoch {best_epoch})')
        ax1.scatter([best_epoch], [best_val], color='gold', s=200, zorder=5, marker='*')
        
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training & Validation Loss Over Time', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Loss Reduction (%)
        ax2 = axes[0, 1]
        initial_train = self.training_data.iloc[0]['train_loss']
        initial_val = self.training_data.iloc[0]['val_loss']
        
        train_reduction = ((initial_train - self.training_data['train_loss']) / initial_train) * 100
        val_reduction = ((initial_val - self.training_data['val_loss']) / initial_val) * 100
        
        ax2.plot(self.training_data['epoch'], train_reduction, 'b-o', 
                label='Train Loss Reduction', linewidth=2, markersize=6)
        ax2.plot(self.training_data['epoch'], val_reduction, 'r-s', 
                label='Val Loss Reduction', linewidth=2, markersize=6)
        ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss Reduction (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Loss Improvement from Initial', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Overfitting Check (Val/Train Ratio)
        ax3 = axes[1, 0]
        loss_ratio = self.training_data['val_loss'] / self.training_data['train_loss']
        ax3.plot(self.training_data['epoch'], loss_ratio, 'purple', linewidth=2.5)
        ax3.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Perfect (Ratio=1)')
        ax3.axhline(1.5, color='orange', linestyle='--', alpha=0.5, label='Moderate Overfitting')
        ax3.axhline(2.0, color='red', linestyle='--', alpha=0.5, label='High Overfitting')
        ax3.fill_between(self.training_data['epoch'], 0, 1, alpha=0.1, color='green')
        ax3.fill_between(self.training_data['epoch'], 1, 1.5, alpha=0.1, color='yellow')
        ax3.fill_between(self.training_data['epoch'], 1.5, loss_ratio.max(), alpha=0.1, color='red')
        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Val Loss / Train Loss', fontsize=12, fontweight='bold')
        ax3.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, min(loss_ratio.max() * 1.1, 5)])
        
        # 4. Loss Distribution Box Plot
        ax4 = axes[1, 1]
        data_to_plot = [self.training_data['train_loss'], self.training_data['val_loss']]
        bp = ax4.boxplot(data_to_plot, labels=['Training Loss', 'Validation Loss'],
                        patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax4.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
        ax4.set_title('Loss Distribution Across All Epochs', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
        
    def generate_report(self, save_path='training_report.csv'):
        """Generate detailed CSV report"""
        if self.training_data is None:
            print("No data to generate report")
            return
        
        # Add calculated metrics
        report = self.training_data.copy()
        report['loss_ratio'] = report['val_loss'] / report['train_loss']
        report['overfitting'] = report['loss_ratio'].apply(
            lambda x: 'High' if x > 2.0 else 'Moderate' if x > 1.5 else 'Low' if x > 1.0 else 'None'
        )
        report['is_best'] = report['val_loss'] == report['val_loss'].min()
        
        # Add cumulative improvement
        initial_val = report.iloc[0]['val_loss']
        report['val_improvement_pct'] = ((initial_val - report['val_loss']) / initial_val) * 100
        
        report.to_csv(save_path, index=False)
        print(f"✓ Saved detailed report to {save_path}")
        
        return report

def main():
    """Main analysis function"""
    print("="*70)
    print("TFT TRAINING RESULTS ANALYZER")
    print("="*70 + "\n")
    
    # Initialize analyzer
    analyzer = TrainingAnalyzer('training.log')
    
    # Parse log
    analyzer.parse_log()
    
    # Print summary
    analyzer.print_summary()
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    analyzer.plot_training_curves('training_curves.png')
    
    # Generate report
    print("\n📋 Generating detailed report...")
    report = analyzer.generate_report('training_report.csv')
    
    print("\n" + "="*70)
    print("WHAT YOU HAVE:")
    print("="*70)
    print("\n✅ TRAINED MODEL FILES:")
    print("   • tft-epoch01-3.1130.pth (initial best)")
    print("   • tft-epoch02-2.6132.pth (BEST MODEL - use this one!)")
    
    print("\n✅ OUTPUTS GENERATED:")
    print("   • training_curves.png    - Visual analysis of training")
    print("   • training_report.csv    - Detailed epoch-by-epoch data")
    
    print("\n📈 NEXT STEPS:")
    print("   1. Use 'tft-epoch02-2.6132.pth' for predictions")
    print("   2. Review 'training_curves.png' for training quality")
    print("   3. Check for overfitting in the ratio plot")
    print("   4. If validation loss is high, consider:")
    print("      - Collecting more training data")
    print("      - Adjusting hyperparameters (learning rate, dropout)")
    print("      - Feature engineering")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
