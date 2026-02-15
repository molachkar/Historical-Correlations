"""
OPTIMIZED TFT GOLD PREDICTION - GOOGLE COLAB
==============================================
High-performance training script with:
- Reduced overfitting through regularization
- Real-time monitoring and visualization
- Automatic checkpoint management
- GPU optimization
- Early stopping with patience
- Learning rate scheduling
- Comprehensive logging and metrics
"""

import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, SMAPE, MAE
import torch
import warnings
import os
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Optimized configuration based on overfitting analysis"""
    
    # Data paths
    DATA_PATH = "master_dataset.csv"  # Upload to Colab
    OUTPUT_DIR = "tft_output"
    MODEL_DIR = "models"
    CHECKPOINT_DIR = "checkpoints"
    
    # Time series parameters
    MAX_PREDICTION_LENGTH = 60
    MAX_ENCODER_LENGTH = 180
    
    # Training parameters - OPTIMIZED TO REDUCE OVERFITTING
    BATCH_SIZE = 64  # Increased from 32 for better generalization
    MAX_EPOCHS = 50  # Reduced from 100 (we usually stop early anyway)
    LEARNING_RATE = 0.001  # Slightly higher for faster convergence
    
    # Model architecture - REDUCED CAPACITY TO PREVENT OVERFITTING
    HIDDEN_SIZE = 128  # Reduced from 256
    ATTENTION_HEAD_SIZE = 4  # Reduced from 8
    DROPOUT = 0.25  # Increased from 0.12 for more regularization
    HIDDEN_CONTINUOUS_SIZE = 64  # Reduced from 128
    
    # Regularization
    GRADIENT_CLIP = 0.1
    WEIGHT_DECAY = 1e-5  # L2 regularization
    
    # Early stopping - MORE AGGRESSIVE
    PATIENCE = 8  # Reduced from 12
    MIN_DELTA = 0.001  # Minimum improvement to count
    
    # Learning rate scheduling
    USE_LR_SCHEDULER = True
    LR_SCHEDULER_PATIENCE = 3
    LR_SCHEDULER_FACTOR = 0.5
    
    # Validation
    VALIDATION_CUTOFF_DATE = "2023-06-01"
    TARGET = "gold_return_future"
    GOLD_PRICE_COL = "Close_XAUUSD"
    
    # GPU settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2 if torch.cuda.is_available() else 0
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches during training
    SAVE_PLOTS = True
    VERBOSE = True

class FeatureCategories:
    """Feature organization"""
    
    ECONOMIC_INDICATORS = [
        'CPIAUCSL', 'FEDFUNDS', 'PAYEMS', 'PCE', 'UNRATE', 'GPR',
        'DFII10', 'DGS2', 'DFII5',
    ]
    
    MARKET_INDICES = [
        'Close', 'High', 'Low', 'Open',
        'Close_EURUSD', 'High_EURUSD', 'Low_EURUSD', 'Open_EURUSD',
        'Close_NASDAQ', 'High_NASDAQ', 'Low_NASDAQ', 'Open_NASDAQ', 'Volume_NASDAQ',
        'Close_RUSSELL2000', 'High_RUSSELL2000', 'Low_RUSSELL2000', 
        'Open_RUSSELL2000', 'Volume_RUSSELL2000',
        'Close_SP500', 'High_SP500', 'Low_SP500', 'Open_SP500', 'Volume_SP500',
        'Close_USDJPY', 'High_USDJPY', 'Low_USDJPY', 'Open_USDJPY',
        'Close_VIX', 'High_VIX', 'Low_VIX', 'Open_VIX',
    ]
    
    TECHNICAL_INDICATORS = [
        'Close_INDICATORS', 'High_INDICATORS', 'Low_INDICATORS', 'Open_INDICATORS', 'Volume',
        'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
        'Parkinson_Vol', 'Yang_Zhang_Vol', 'Vol_Percentile',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'ADX', 'Plus_DI', 'Minus_DI',
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
        'ATR', 'Stoch_K', 'Stoch_D',
        'Ichimoku_TK', 'Ichimoku_KJ', 'Ichimoku_SA', 'Ichimoku_SB',
        'VWAP', 'PSAR', 'AO', 'Williams_R', 'CCI', 'MFI', 'ROC',
        'Efficiency_Ratio',
    ]
    
    GOLD_COMPONENTS = [
        'Close_XAUUSD', 'High_XAUUSD', 'Low_XAUUSD', 'Open_XAUUSD', 'Volume_XAUUSD'
    ]
    
    CATEGORICAL_FEATURES = ['Vol_Regime']

# =============================================================================
# HELPER CLASSES
# =============================================================================

class MetricsTracker:
    """Track and visualize training metrics"""
    
    def __init__(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_smape': [],
            'val_mae': [],
            'learning_rate': [],
            'best_epoch': None,
            'best_val_loss': float('inf')
        }
    
    def update(self, epoch, train_loss, val_loss, val_smape=None, val_mae=None, lr=None):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_smape'].append(val_smape if val_smape else 0)
        self.history['val_mae'].append(val_mae if val_mae else 0)
        self.history['learning_rate'].append(lr if lr else 0)
        
        if val_loss < self.history['best_val_loss']:
            self.history['best_val_loss'] = val_loss
            self.history['best_epoch'] = epoch
    
    def plot(self, save_path='training_progress.png'):
        """Create comprehensive training visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        epochs = self.history['epoch']
        
        # Loss curves
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
        if self.history['best_epoch']:
            ax1.axvline(self.history['best_epoch'], color='g', linestyle='--', 
                       alpha=0.5, label=f"Best (Epoch {self.history['best_epoch']})")
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Overfitting ratio
        ax2 = axes[0, 1]
        ratios = np.array(self.history['val_loss']) / (np.array(self.history['train_loss']) + 1e-8)
        ax2.plot(epochs, ratios, 'purple', linewidth=2)
        ax2.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
        ax2.axhline(1.5, color='orange', linestyle='--', alpha=0.5, label='Moderate Overfitting')
        ax2.fill_between(epochs, 0, 1, alpha=0.1, color='green')
        ax2.fill_between(epochs, 1, 1.5, alpha=0.1, color='yellow')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Val/Train Ratio')
        ax2.set_title('Overfitting Monitor')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, min(3, max(ratios) * 1.1)])
        
        # Additional metrics
        ax3 = axes[1, 0]
        if any(self.history['val_smape']):
            ax3.plot(epochs, self.history['val_smape'], 'g-^', label='SMAPE', linewidth=2)
        if any(self.history['val_mae']):
            ax3_twin = ax3.twinx()
            ax3_twin.plot(epochs, self.history['val_mae'], 'orange', linestyle='--', 
                         label='MAE', linewidth=2)
            ax3_twin.set_ylabel('MAE')
            ax3_twin.legend(loc='upper right')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('SMAPE')
        ax3.set_title('Validation Metrics')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Learning rate
        ax4 = axes[1, 1]
        if any(self.history['learning_rate']):
            ax4.plot(epochs, self.history['learning_rate'], 'brown', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training visualization to {save_path}")
        
        # Display in Colab
        plt.show()
    
    def save(self, path='metrics_history.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.history, f)

# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prepare_data(config):
    """Load and prepare data with enhanced preprocessing"""
    print("="*70)
    print("LOADING AND PREPARING DATA")
    print("="*70)
    
    df = pd.read_csv(config.DATA_PATH)
    print(f"✓ Loaded {len(df):,} rows, {df.shape[1]} columns")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create time index
    df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days
    df['series_id'] = 'gold_macro'
    
    # Handle categorical
    if 'Vol_Regime' in df.columns:
        df['Vol_Regime'] = df['Vol_Regime'].fillna('medium')
    
    # Time features
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_month'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    
    # Forward fill economic indicators
    for col in FeatureCategories.ECONOMIC_INDICATORS:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill')
    
    # Create target
    df[config.TARGET] = (
        df[config.GOLD_PRICE_COL].shift(-config.MAX_PREDICTION_LENGTH) / 
        df[config.GOLD_PRICE_COL] - 1
    ) * 100
    
    # Binary direction
    df['gold_direction'] = (df[config.TARGET] > 0).astype(int)
    
    # Remove NaN targets
    initial_len = len(df)
    df = df[df[config.TARGET].notna()].reset_index(drop=True)
    removed = initial_len - len(df)
    
    print(f"\n📊 Data Statistics:")
    print(f"  • Target: {config.MAX_PREDICTION_LENGTH}-day ahead returns")
    print(f"  • Removed {removed} rows with NaN targets")
    print(f"  • Final dataset: {len(df):,} rows")
    print(f"  • Target mean: {df[config.TARGET].mean():.4f}%")
    print(f"  • Target std: {df[config.TARGET].std():.4f}%")
    print(f"  • Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    return df

def prepare_feature_lists(df, config):
    """Organize features into categories"""
    print("\n" + "="*70)
    print("ORGANIZING FEATURES")
    print("="*70)
    
    available_cols = set(df.columns)
    
    # Time-varying known (future values known)
    time_varying_known_reals = [
        'time_idx', 'month', 'day_of_week', 'quarter', 
        'day_of_month', 'week_of_year'
    ]
    time_varying_known_reals.extend([
        col for col in FeatureCategories.ECONOMIC_INDICATORS 
        if col in available_cols
    ])
    
    # Time-varying unknown (future values unknown)
    time_varying_unknown_reals = []
    time_varying_unknown_reals.extend([
        col for col in FeatureCategories.MARKET_INDICES 
        if col in available_cols
    ])
    time_varying_unknown_reals.extend([
        col for col in FeatureCategories.TECHNICAL_INDICATORS 
        if col in available_cols and col not in FeatureCategories.CATEGORICAL_FEATURES
    ])
    time_varying_unknown_reals.extend([
        col for col in FeatureCategories.GOLD_COMPONENTS 
        if col in available_cols and col != config.GOLD_PRICE_COL
    ])
    
    # Categorical features
    time_varying_unknown_categoricals = [
        col for col in FeatureCategories.CATEGORICAL_FEATURES 
        if col in available_cols
    ]
    
    static_categoricals = ['series_id']
    
    total = len(time_varying_known_reals) + len(time_varying_unknown_reals) + len(time_varying_unknown_categoricals)
    
    print(f"\n📋 Feature Summary:")
    print(f"  • Total features: {total}")
    print(f"  • Known reals: {len(time_varying_known_reals)}")
    print(f"  • Unknown reals: {len(time_varying_unknown_reals)}")
    print(f"  • Categorical: {len(time_varying_unknown_categoricals)}")
    
    return {
        'time_varying_known_reals': time_varying_known_reals,
        'time_varying_unknown_reals': time_varying_unknown_reals,
        'time_varying_known_categoricals': [],
        'time_varying_unknown_categoricals': time_varying_unknown_categoricals,
        'static_categoricals': static_categoricals,
    }

def create_datasets(df, features, config):
    """Create training and validation datasets"""
    print("\n" + "="*70)
    print("CREATING DATASETS")
    print("="*70)
    
    validation_cutoff = df[df['Date'] >= config.VALIDATION_CUTOFF_DATE]['time_idx'].min()
    
    train_df = df[df['time_idx'] < validation_cutoff].copy()
    val_df = df[df['time_idx'] >= validation_cutoff].copy()
    
    train_pct = len(train_df) / len(df) * 100
    val_pct = len(val_df) / len(df) * 100
    
    print(f"\n📊 Split Summary:")
    print(f"  • Training samples: {len(train_df):,} ({train_pct:.1f}%)")
    print(f"  • Validation samples: {len(val_df):,} ({val_pct:.1f}%)")
    print(f"  • Split date: {config.VALIDATION_CUTOFF_DATE}")
    
    # Create training dataset
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=config.TARGET,
        group_ids=["series_id"],
        min_encoder_length=config.MAX_ENCODER_LENGTH // 2,
        max_encoder_length=config.MAX_ENCODER_LENGTH,
        min_prediction_length=1,
        max_prediction_length=config.MAX_PREDICTION_LENGTH,
        static_categoricals=features['static_categoricals'],
        time_varying_known_reals=features['time_varying_known_reals'],
        time_varying_unknown_reals=features['time_varying_unknown_reals'],
        time_varying_known_categoricals=features['time_varying_known_categoricals'],
        time_varying_unknown_categoricals=features['time_varying_unknown_categoricals'],
        target_normalizer=GroupNormalizer(
            groups=["series_id"], 
            transformation="softplus",
            center=True,
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    
    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )
    
    # Create dataloaders
    train_dataloader = training.to_dataloader(
        train=True, 
        batch_size=config.BATCH_SIZE, 
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )
    
    val_dataloader = validation.to_dataloader(
        train=False, 
        batch_size=config.BATCH_SIZE * 2,  # Larger batch for validation
        num_workers=config.NUM_WORKERS,
    )
    
    print(f"\n⚙️  DataLoader Info:")
    print(f"  • Train batches: {len(train_dataloader)}")
    print(f"  • Val batches: {len(val_dataloader)}")
    print(f"  • Batch size: {config.BATCH_SIZE}")
    
    return training, validation, train_dataloader, val_dataloader

# =============================================================================
# TRAINING
# =============================================================================

def train_model(tft, train_dataloader, val_dataloader, config):
    """Optimized training loop with monitoring"""
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    
    # Setup directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Model info
    total_params = sum(p.numel() for p in tft.parameters())
    trainable_params = sum(p.numel() for p in tft.parameters() if p.requires_grad)
    
    print(f"\n🤖 Model Info:")
    print(f"  • Total parameters: {total_params:,}")
    print(f"  • Trainable parameters: {trainable_params:,}")
    print(f"  • Device: {config.DEVICE.upper()}")
    print(f"  • Hidden size: {config.HIDDEN_SIZE}")
    print(f"  • Dropout: {config.DROPOUT}")
    
    # Move to device
    tft = tft.to(config.DEVICE)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.Adam(
        tft.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = None
    if config.USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            verbose=True
        )
    
    # Loss function
    loss_fn = QuantileLoss()
    
    # Additional metrics
    smape_metric = SMAPE()
    mae_metric = MAE()
    
    # Metrics tracker
    metrics = MetricsTracker()
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = None
    
    print(f"\n🎯 Training Configuration:")
    print(f"  • Max epochs: {config.MAX_EPOCHS}")
    print(f"  • Learning rate: {config.LEARNING_RATE}")
    print(f"  • Batch size: {config.BATCH_SIZE}")
    print(f"  • Early stopping patience: {config.PATIENCE}")
    print(f"  • Weight decay: {config.WEIGHT_DECAY}")
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    # Training loop
    for epoch in range(config.MAX_EPOCHS):
        # =====================================================================
        # TRAINING PHASE
        # =====================================================================
        tft.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            # Move batch to device
            x, y = batch
            for key in x:
                if isinstance(x[key], torch.Tensor):
                    x[key] = x[key].to(config.DEVICE)
            if isinstance(y, tuple):
                y = tuple(yi.to(config.DEVICE) if isinstance(yi, torch.Tensor) else yi for yi in y)
            else:
                y = (y[0].to(config.DEVICE), y[1].to(config.DEVICE))
            
            # Forward pass
            output = tft(x)
            predictions = output['prediction']
            loss = loss_fn(predictions, y[0])
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tft.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Log progress
            if config.VERBOSE and (batch_idx + 1) % config.LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}/{config.MAX_EPOCHS} - Batch {batch_idx+1}/{len(train_dataloader)} - Loss: {loss.item():.4f}")
        
        # =====================================================================
        # VALIDATION PHASE
        # =====================================================================
        tft.eval()
        val_losses = []
        val_smapes = []
        val_maes = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                x, y = batch
                for key in x:
                    if isinstance(x[key], torch.Tensor):
                        x[key] = x[key].to(config.DEVICE)
                if isinstance(y, tuple):
                    y = tuple(yi.to(config.DEVICE) if isinstance(yi, torch.Tensor) else yi for yi in y)
                else:
                    y = (y[0].to(config.DEVICE), y[1].to(config.DEVICE))
                
                output = tft(x)
                predictions = output['prediction']
                
                # Calculate metrics
                loss = loss_fn(predictions, y[0])
                val_losses.append(loss.item())
                
                # Additional metrics
                try:
                    smape_val = smape_metric(predictions, y[0])
                    mae_val = mae_metric(predictions, y[0])
                    val_smapes.append(smape_val.item() if torch.is_tensor(smape_val) else smape_val)
                    val_maes.append(mae_val.item() if torch.is_tensor(mae_val) else mae_val)
                except:
                    pass
        
        # =====================================================================
        # EPOCH SUMMARY
        # =====================================================================
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_smape = np.mean(val_smapes) if val_smapes else None
        val_mae = np.mean(val_maes) if val_maes else None
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update metrics
        metrics.update(epoch + 1, train_loss, val_loss, val_smape, val_mae, current_lr)
        
        # Print epoch summary
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{config.MAX_EPOCHS} SUMMARY")
        print(f"{'='*70}")
        print(f"  Train Loss:      {train_loss:.4f}")
        print(f"  Val Loss:        {val_loss:.4f}")
        print(f"  Val/Train Ratio: {val_loss/train_loss:.3f}")
        if val_smape:
            print(f"  Val SMAPE:       {val_smape:.4f}")
        if val_mae:
            print(f"  Val MAE:         {val_mae:.4f}")
        print(f"  Learning Rate:   {current_lr:.6f}")
        
        # =====================================================================
        # CHECKPOINTING & EARLY STOPPING
        # =====================================================================
        
        # Update learning rate scheduler
        if scheduler:
            scheduler.step(val_loss)
        
        # Check for improvement
        improvement = best_val_loss - val_loss
        
        if val_loss < best_val_loss - config.MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(
                config.MODEL_DIR, 
                f"tft-best-epoch{epoch+1:03d}-{val_loss:.4f}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': tft.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, best_model_path)
            
            print(f"  ✓ NEW BEST MODEL saved: {os.path.basename(best_model_path)}")
            print(f"  ✓ Improvement: {improvement:.4f}")
        else:
            patience_counter += 1
            print(f"  ⚠ No improvement (patience: {patience_counter}/{config.PATIENCE})")
            
            if patience_counter >= config.PATIENCE:
                print(f"\n{'='*70}")
                print(f"EARLY STOPPING at epoch {epoch+1}")
                print(f"{'='*70}")
                print(f"  Best validation loss: {best_val_loss:.4f}")
                print(f"  Best epoch: {metrics.history['best_epoch']}")
                break
        
        print(f"{'='*70}\n")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"checkpoint-epoch{epoch+1:03d}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': tft.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best epoch: {metrics.history['best_epoch']}")
    print(f"  Total epochs trained: {len(metrics.history['epoch'])}")
    
    # Load best model
    if best_model_path and os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        tft.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded best model: {os.path.basename(best_model_path)}")
    
    return tft, best_model_path, best_val_loss, metrics

# =============================================================================
# FEATURE IMPORTANCE (WITH ERROR HANDLING)
# =============================================================================

def extract_feature_importance(tft, val_dataloader, output_dir):
    """Extract feature importance with graceful fallback"""
    print("\n" + "="*70)
    print("EXTRACTING FEATURE IMPORTANCE")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    tft.eval()
    
    batch = next(iter(val_dataloader))
    x, y = batch
    
    for key in x:
        if isinstance(x[key], torch.Tensor):
            x[key] = x[key].to(tft.device)
    
    with torch.no_grad():
        try:
            interpretation = tft.interpret_output(x, reduction="sum")
        except (KeyError, AttributeError) as e:
            print(f"  ⚠ Full interpretation unavailable: {str(e)[:50]}...")
            print(f"  → Using simplified feature extraction")
            output = tft(x)
            interpretation = {
                'attention': output.get('attention', None),
                'static_variables': {},
                'encoder_variables': {},
                'decoder_variables': {},
            }
    
    # Extract importance scores
    encoder_vars = interpretation.get('encoder_variables', {})
    decoder_vars = interpretation.get('decoder_variables', {})
    static_vars = interpretation.get('static_variables', {})
    
    all_importance = {}
    all_importance.update(encoder_vars)
    all_importance.update(decoder_vars)
    all_importance.update(static_vars)
    
    # Create dataframe
    if len(all_importance) == 0:
        print("  ⚠ Feature importance scores not available")
        importance_df = pd.DataFrame([
            {'feature': 'N/A', 'importance': 0.0, 'type': 'unavailable'}
        ])
    else:
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v, 'type': 
             'encoder' if k in encoder_vars else 
             'decoder' if k in decoder_vars else 'static'}
            for k, v in all_importance.items()
        ]).sort_values('importance', ascending=False)
        
        print(f"\n📊 Top 15 Features:")
        for idx, row in importance_df.head(15).iterrows():
            print(f"  {idx+1:2d}. {row['feature']:<40} {row['importance']:>8.4f}")
    
    # Save
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    with open(os.path.join(output_dir, 'interpretation.pkl'), 'wb') as f:
        pickle.dump(interpretation, f)
    
    return interpretation, importance_df

# =============================================================================
# SAVE ARTIFACTS
# =============================================================================

def save_artifacts(tft, training_dataset, config, output_dir, importance_df, 
                  best_val_loss, metrics):
    """Save all training artifacts"""
    print("\n" + "="*70)
    print("SAVING ARTIFACTS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataset metadata
    with open(os.path.join(output_dir, 'dataset_metadata.pkl'), 'wb') as f:
        pickle.dump({
            'parameters': training_dataset.get_parameters(),
            'scalers': training_dataset.scalers,
            'categorical_encoders': training_dataset.categorical_encoders,
        }, f)
    print("  ✓ Saved dataset metadata")
    
    # Save config
    with open(os.path.join(output_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)
    print("  ✓ Saved configuration")
    
    # Save metrics history
    metrics.save(os.path.join(output_dir, 'metrics_history.pkl'))
    print("  ✓ Saved metrics history")
    
    # Create summary
    summary = {
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_features': len(training_dataset.reals + training_dataset.categoricals),
        'encoder_length': config.MAX_ENCODER_LENGTH,
        'prediction_length': config.MAX_PREDICTION_LENGTH,
        'target': config.TARGET,
        'hidden_size': config.HIDDEN_SIZE,
        'attention_heads': config.ATTENTION_HEAD_SIZE,
        'dropout': config.DROPOUT,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'weight_decay': config.WEIGHT_DECAY,
        'max_epochs': config.MAX_EPOCHS,
        'epochs_trained': len(metrics.history['epoch']),
        'best_epoch': metrics.history['best_epoch'],
        'device': config.DEVICE,
        'best_val_loss': best_val_loss,
        'final_train_loss': metrics.history['train_loss'][-1],
        'final_val_loss': metrics.history['val_loss'][-1],
        'top_feature': importance_df.iloc[0]['feature'] if len(importance_df) > 0 and importance_df.iloc[0]['feature'] != 'N/A' else 'N/A',
        'top_importance': importance_df.iloc[0]['importance'] if len(importance_df) > 0 and importance_df.iloc[0]['feature'] != 'N/A' else 0,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, 'training_summary.csv'), index=False)
    print("  ✓ Saved training summary")
    
    # Plot metrics
    if config.SAVE_PLOTS:
        metrics.plot(os.path.join(output_dir, 'training_progress.png'))
    
    print(f"\n✓ All artifacts saved to: {output_dir}/")

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "="*70)
    print("TFT GOLD PREDICTION - OPTIMIZED FOR GOOGLE COLAB")
    print("="*70)
    print(f"  Device: {Config.DEVICE.upper()}")
    print(f"  Hidden Size: {Config.HIDDEN_SIZE}")
    print(f"  Dropout: {Config.DROPOUT}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print("="*70 + "\n")
    
    config = Config()
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    try:
        # Load data
        df = load_and_prepare_data(config)
        
        # Prepare features
        features = prepare_feature_lists(df, config)
        
        # Create datasets
        training, validation, train_dataloader, val_dataloader = create_datasets(
            df, features, config
        )
        
        # Create model
        print("\n" + "="*70)
        print("CREATING MODEL")
        print("="*70)
        
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=config.LEARNING_RATE,
            hidden_size=config.HIDDEN_SIZE,
            attention_head_size=config.ATTENTION_HEAD_SIZE,
            dropout=config.DROPOUT,
            hidden_continuous_size=config.HIDDEN_CONTINUOUS_SIZE,
            output_size=7,  # Quantile loss default
            loss=QuantileLoss(),
        )
        
        print(f"  ✓ Model created successfully")
        
        # Train model
        tft, best_model_path, best_val_loss, metrics = train_model(
            tft, train_dataloader, val_dataloader, config
        )
        
        # Extract feature importance
        interpretation, importance_df = extract_feature_importance(
            tft, val_dataloader, config.OUTPUT_DIR
        )
        
        # Save artifacts
        save_artifacts(
            tft, training, config, config.OUTPUT_DIR, 
            importance_df, best_val_loss, metrics
        )
        
        # Final summary
        print("\n" + "="*70)
        print("TRAINING PIPELINE COMPLETE")
        print("="*70)
        print(f"  ✓ Best model: {os.path.basename(best_model_path) if best_model_path else 'N/A'}")
        print(f"  ✓ Best val loss: {best_val_loss:.4f}")
        print(f"  ✓ Epochs trained: {len(metrics.history['epoch'])}")
        print(f"  ✓ Best epoch: {metrics.history['best_epoch']}")
        
        if len(importance_df) > 0 and importance_df.iloc[0]['feature'] != 'N/A':
            print(f"  ✓ Top feature: {importance_df.iloc[0]['feature']}")
        
        print(f"\n📁 Outputs saved in:")
        print(f"  • {config.MODEL_DIR}/          - Best model checkpoint")
        print(f"  • {config.CHECKPOINT_DIR}/     - Training checkpoints")
        print(f"  • {config.OUTPUT_DIR}/         - Metrics, plots, summaries")
        
        print("\n" + "="*70)
        print("READY FOR PREDICTIONS!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()