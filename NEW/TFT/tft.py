"""
TEMPORAL FUSION TRANSFORMER (TFT) TRAINING PIPELINE
Gold Market Causal Intelligence Engine - OPTIMIZED VERSION

Tailored specifically for your master dataset structure with:
- 81 columns of macro + market + technical indicator data
- Mixed frequency data (daily prices + monthly economic)
- Optimized for 8GB RAM overnight training
"""

import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import torch
import warnings
import os
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR YOUR HARDWARE
# ============================================================================

class Config:
    """Training configuration - Safe overnight settings for 8GB RAM"""
    
    # Paths
    DATA_PATH = "master_dataset.csv"
    OUTPUT_DIR = "tft_output"
    MODEL_DIR = "models"
    
    # Time series parameters (Balanced for deep insights)
    MAX_PREDICTION_LENGTH = 45   # Predict 45 days (1.5 months) ahead
    MAX_ENCODER_LENGTH = 120     # Look back 120 days (4 months context)
    
    # Training parameters (Safe for 8GB RAM)
    BATCH_SIZE = 24              # Safe batch size
    MAX_EPOCHS = 80              # Will auto-stop if converged
    LEARNING_RATE = 0.001        # Stable learning rate
    HIDDEN_SIZE = 192            # Deep model, won't crash
    ATTENTION_HEAD_SIZE = 6      # Good attention coverage
    DROPOUT = 0.15               # Prevent overfitting
    
    # Validation split
    VALIDATION_CUTOFF_DATE = "2023-06-01"  # 18 months validation
    
    # Target configuration
    TARGET = "gold_return_future"
    GOLD_PRICE_COL = "Close_XAUUSD"  # Your gold close price
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# FEATURE CATEGORIZATION - BASED ON YOUR EXACT COLUMNS
# ============================================================================

class FeatureCategories:
    """
    Intelligent feature categorization based on your data structure
    """
    
    # Economic indicators (monthly, slow-moving, known in advance)
    ECONOMIC_INDICATORS = [
        'CPIAUCSL',      # Consumer Price Index
        'FEDFUNDS',      # Federal Funds Rate
        'PAYEMS',        # Total Nonfarm Payrolls
        'PCE',           # Personal Consumption Expenditures
        'UNRATE',        # Unemployment Rate
        'GPR',           # Geopolitical Risk Index
        'DFII10',        # 10-Year TIPS
        'DGS2',          # 2-Year Treasury
        'DFII5',         # 5-Year TIPS
    ]
    
    # Market indices (daily, unknown future)
    MARKET_INDICES = [
        # DXY (Dollar Index)
        'Close', 'High', 'Low', 'Open',
        
        # EUR/USD
        'Close_EURUSD', 'High_EURUSD', 'Low_EURUSD', 'Open_EURUSD',
        
        # NASDAQ
        'Close_NASDAQ', 'High_NASDAQ', 'Low_NASDAQ', 'Open_NASDAQ', 'Volume_NASDAQ',
        
        # Russell 2000
        'Close_RUSSELL2000', 'High_RUSSELL2000', 'Low_RUSSELL2000', 
        'Open_RUSSELL2000', 'Volume_RUSSELL2000',
        
        # S&P 500
        'Close_SP500', 'High_SP500', 'Low_SP500', 'Open_SP500', 'Volume_SP500',
        
        # USD/JPY
        'Close_USDJPY', 'High_USDJPY', 'Low_USDJPY', 'Open_USDJPY',
        
        # VIX (Volatility)
        'Close_VIX', 'High_VIX', 'Low_VIX', 'Open_VIX',
    ]
    
    # Technical indicators from your INDICATORS.csv
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
    
    # Gold price components (target-related)
    GOLD_COMPONENTS = [
        'Close_XAUUSD', 'High_XAUUSD', 'Low_XAUUSD', 'Open_XAUUSD', 'Volume_XAUUSD'
    ]
    
    # Categorical features (regime indicators)
    CATEGORICAL_FEATURES = [
        'Vol_Regime',  # low/medium/high
    ]


# ============================================================================
# STEP 1: DATA PREPARATION
# ============================================================================

def load_and_prepare_data(config):
    """
    Load master dataset and prepare for TFT with proper handling
    """
    print("="*70)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*70)
    
    # Load master dataset
    print(f"\nLoading data from: {config.DATA_PATH}")
    df = pd.read_csv(config.DATA_PATH)
    
    print(f"✓ Loaded {len(df)} rows")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Convert date to datetime (handle both formats)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Create time index (required by TFT)
    df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days
    
    # Create group column (single time series)
    df['series_id'] = 'gold_macro'
    
    # Handle missing values in Vol_Regime (fill with 'medium')
    if 'Vol_Regime' in df.columns:
        df['Vol_Regime'] = df['Vol_Regime'].fillna('medium')
    
    # Create temporal features (known in advance)
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_month'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    
    # Create lagged economic indicators (they're released with delay)
    for col in FeatureCategories.ECONOMIC_INDICATORS:
        if col in df.columns:
            # Forward fill monthly data (simulate real-world lag)
            df[col] = df[col].fillna(method='ffill')
    
    # Create target variable: Gold returns N days ahead
    df[config.TARGET] = (
        df[config.GOLD_PRICE_COL].shift(-config.MAX_PREDICTION_LENGTH) / 
        df[config.GOLD_PRICE_COL] - 1
    ) * 100  # Convert to percentage returns
    
    # Create additional target: Direction (up/down)
    df['gold_direction'] = (df[config.TARGET] > 0).astype(int)
    
    # Remove rows with NaN in target (at the end)
    initial_len = len(df)
    df = df[df[config.TARGET].notna()].reset_index(drop=True)
    removed = initial_len - len(df)
    
    print(f"\n✓ Created target variable: {config.TARGET}")
    print(f"  Target: {config.MAX_PREDICTION_LENGTH}-day ahead gold returns")
    print(f"  Removed {removed} rows with missing target")
    print(f"  Final dataset size: {len(df)} rows")
    
    # Data quality check
    print(f"\n✓ Data quality check:")
    print(f"  Target mean: {df[config.TARGET].mean():.4f}%")
    print(f"  Target std: {df[config.TARGET].std():.4f}%")
    print(f"  Target range: [{df[config.TARGET].min():.2f}%, {df[config.TARGET].max():.2f}%]")
    print(f"  Positive days: {(df[config.TARGET] > 0).sum()} ({(df[config.TARGET] > 0).mean()*100:.1f}%)")
    
    return df


def prepare_feature_lists(df, config):
    """
    Prepare feature lists for TFT based on data availability
    """
    print("\n" + "="*70)
    print("STEP 2: CATEGORIZING FEATURES FOR TFT")
    print("="*70)
    
    available_cols = set(df.columns)
    
    # Time-varying known reals (we know these in advance)
    time_varying_known_reals = [
        'time_idx', 'month', 'day_of_week', 'quarter', 
        'day_of_month', 'week_of_year'
    ]
    
    # Add economic indicators (released monthly, known in advance with lag)
    time_varying_known_reals.extend([
        col for col in FeatureCategories.ECONOMIC_INDICATORS 
        if col in available_cols
    ])
    
    # Time-varying unknown reals (market data - unknown in future)
    time_varying_unknown_reals = []
    
    # Add market indices
    time_varying_unknown_reals.extend([
        col for col in FeatureCategories.MARKET_INDICES 
        if col in available_cols
    ])
    
    # Add technical indicators
    time_varying_unknown_reals.extend([
        col for col in FeatureCategories.TECHNICAL_INDICATORS 
        if col in available_cols and col not in FeatureCategories.CATEGORICAL_FEATURES
    ])
    
    # Add gold components (except target column)
    time_varying_unknown_reals.extend([
        col for col in FeatureCategories.GOLD_COMPONENTS 
        if col in available_cols and col != config.GOLD_PRICE_COL
    ])
    
    # Time-varying unknown categoricals
    time_varying_unknown_categoricals = [
        col for col in FeatureCategories.CATEGORICAL_FEATURES 
        if col in available_cols
    ]
    
    # Static categoricals
    static_categoricals = ['series_id']
    
    print(f"\n✓ Feature categorization complete:")
    print(f"  Time-varying KNOWN reals: {len(time_varying_known_reals)} features")
    print(f"    (Economic indicators + calendar features)")
    print(f"  Time-varying UNKNOWN reals: {len(time_varying_unknown_reals)} features")
    print(f"    (Market prices + technical indicators)")
    print(f"  Time-varying UNKNOWN categoricals: {len(time_varying_unknown_categoricals)} features")
    print(f"  Static categoricals: {len(static_categoricals)} features")
    print(f"  Total features: {len(time_varying_known_reals) + len(time_varying_unknown_reals) + len(time_varying_unknown_categoricals)}")
    
    # Print feature breakdown
    print(f"\n  Top economic indicators (known):")
    for feat in time_varying_known_reals[6:12]:  # Skip time features
        print(f"    - {feat}")
    
    print(f"\n  Sample market features (unknown):")
    for feat in time_varying_unknown_reals[:8]:
        print(f"    - {feat}")
    
    return {
        'time_varying_known_reals': time_varying_known_reals,
        'time_varying_unknown_reals': time_varying_unknown_reals,
        'time_varying_known_categoricals': [],
        'time_varying_unknown_categoricals': time_varying_unknown_categoricals,
        'static_categoricals': static_categoricals,
    }


# ============================================================================
# STEP 3: CREATE TIMESERIES DATASET
# ============================================================================

def create_datasets(df, features, config):
    """
    Create training and validation datasets with proper handling
    """
    print("\n" + "="*70)
    print("STEP 3: CREATING TIMESERIES DATASETS")
    print("="*70)
    
    # Split into train and validation
    validation_cutoff = df[df['Date'] >= config.VALIDATION_CUTOFF_DATE]['time_idx'].min()
    
    train_df = df[df['time_idx'] < validation_cutoff].copy()
    val_df = df[df['time_idx'] >= validation_cutoff].copy()
    
    print(f"\nSplitting data:")
    print(f"  Training: {len(train_df)} samples ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"  Validation: {len(val_df)} samples ({val_df['Date'].min()} to {val_df['Date'].max()})")
    print(f"  Split ratio: {len(train_df)/(len(train_df)+len(val_df))*100:.1f}% train")
    
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
        num_workers=0,
        shuffle=True,
    )
    
    val_dataloader = validation.to_dataloader(
        train=False, 
        batch_size=config.BATCH_SIZE * 4, 
        num_workers=0,
    )
    
    print(f"\n✓ Datasets created successfully:")
    print(f"  Training batches: {len(train_dataloader)}")
    print(f"  Validation batches: {len(val_dataloader)}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Encoder length: {config.MAX_ENCODER_LENGTH} days")
    print(f"  Prediction length: {config.MAX_PREDICTION_LENGTH} days")
    
    return training, validation, train_dataloader, val_dataloader


# ============================================================================
# STEP 4: BUILD TFT MODEL
# ============================================================================

def build_tft_model(training_dataset, config):
    """
    Build Temporal Fusion Transformer model with optimal settings
    """
    print("\n" + "="*70)
    print("STEP 4: BUILDING TFT MODEL")
    print("="*70)
    
    # Create output directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Configure trainer with early stopping and checkpointing
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator="gpu" if config.DEVICE == "cuda" else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="min",
                verbose=True,
            ),
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                dirpath=config.MODEL_DIR,
                filename="tft-{epoch:02d}-{val_loss:.4f}",
                save_top_k=3,
                mode="min",
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    # Initialize TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config.LEARNING_RATE,
        hidden_size=config.HIDDEN_SIZE,
        attention_head_size=config.ATTENTION_HEAD_SIZE,
        dropout=config.DROPOUT,
        hidden_continuous_size=config.HIDDEN_SIZE // 2,
        output_size=7,  # 7 quantiles for prediction intervals
        loss=QuantileLoss(),
        optimizer="adam",
        reduce_on_plateau_patience=4,
        logging_metrics=["loss", "MAE", "RMSE"],
    )
    
    print(f"\n✓ TFT Model initialized:")
    print(f"  Hidden size: {config.HIDDEN_SIZE}")
    print(f"  Attention heads: {config.ATTENTION_HEAD_SIZE}")
    print(f"  Dropout: {config.DROPOUT}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Max epochs: {config.MAX_EPOCHS}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Total parameters: {sum(p.numel() for p in tft.parameters()):,}")
    
    print(f"\n  Estimated training time:")
    print(f"    CPU: ~10-12 hours")
    print(f"    GPU: ~45-60 minutes")
    
    return tft, trainer


# ============================================================================
# STEP 5: TRAIN MODEL
# ============================================================================

def train_model(tft, trainer, train_dataloader, val_dataloader):
    """
    Train the TFT model with progress tracking
    """
    print("\n" + "="*70)
    print("STEP 5: TRAINING MODEL")
    print("="*70)
    print("\n🚀 Starting training...")
    print("   (This will take ~10-12 hours on CPU)")
    print("   Training will auto-stop if validation loss stops improving\n")
    
    # Train
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    print("\n✓ Training complete!")
    print(f"  Best model: {trainer.checkpoint_callback.best_model_path}")
    print(f"  Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    print(f"  Total epochs: {trainer.current_epoch}")
    
    return trainer.checkpoint_callback.best_model_path


# ============================================================================
# STEP 6: EXTRACT FEATURE IMPORTANCE
# ============================================================================

def extract_feature_importance(tft, val_dataloader, training_dataset, output_dir):
    """
    Extract feature importance - your "correlation intelligence"
    """
    print("\n" + "="*70)
    print("STEP 6: EXTRACTING CORRELATION INTELLIGENCE")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sample for interpretation
    sample_data = val_dataloader.dataset[:min(100, len(val_dataloader.dataset))]
    
    # Get interpretation
    print("\nComputing feature importance...")
    interpretation = tft.interpret_output(sample_data, reduction="sum")
    
    # Extract variable importance
    encoder_vars = interpretation.get('encoder_variables', {})
    decoder_vars = interpretation.get('decoder_variables', {})
    static_vars = interpretation.get('static_variables', {})
    
    # Combine all importances
    all_importance = {}
    all_importance.update(encoder_vars)
    all_importance.update(decoder_vars)
    all_importance.update(static_vars)
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v, 'type': 'encoder' if k in encoder_vars else 'decoder' if k in decoder_vars else 'static'}
        for k, v in all_importance.items()
    ]).sort_values('importance', ascending=False)
    
    # Save
    importance_df.to_csv(
        os.path.join(output_dir, 'feature_importance.csv'),
        index=False
    )
    
    # Save full interpretation
    with open(os.path.join(output_dir, 'correlation_intelligence.pkl'), 'wb') as f:
        pickle.dump(interpretation, f)
    
    print(f"\n✓ Correlation intelligence extracted!")
    print(f"\n📊 TOP 15 MOST IMPORTANT FEATURES:")
    print("=" * 70)
    for idx, row in importance_df.head(15).iterrows():
        print(f"  {row['feature']:<35} {row['importance']:>8.4f}  [{row['type']}]")
    
    print(f"\n✓ Files saved to: {output_dir}/")
    
    return interpretation, importance_df


# ============================================================================
# STEP 7: SAVE ARTIFACTS
# ============================================================================

def save_training_artifacts(tft, training_dataset, config, output_dir, importance_df):
    """
    Save all necessary artifacts for inference
    """
    print("\n" + "="*70)
    print("STEP 7: SAVING TRAINING ARTIFACTS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataset metadata
    with open(os.path.join(output_dir, 'dataset_metadata.pkl'), 'wb') as f:
        pickle.dump({
            'parameters': training_dataset.get_parameters(),
            'scalers': training_dataset.scalers,
            'categorical_encoders': training_dataset.categorical_encoders,
        }, f)
    
    # Save config
    with open(os.path.join(output_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)
    
    # Create comprehensive training summary
    summary = {
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_features': len(training_dataset.reals + training_dataset.categoricals),
        'encoder_length': config.MAX_ENCODER_LENGTH,
        'prediction_length': config.MAX_PREDICTION_LENGTH,
        'target': config.TARGET,
        'hidden_size': config.HIDDEN_SIZE,
        'attention_heads': config.ATTENTION_HEAD_SIZE,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'max_epochs': config.MAX_EPOCHS,
        'device': config.DEVICE,
        'top_feature': importance_df.iloc[0]['feature'] if len(importance_df) > 0 else 'N/A',
        'top_importance': importance_df.iloc[0]['importance'] if len(importance_df) > 0 else 0,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, 'training_summary.csv'), index=False)
    
    # Save feature lists for inference
    feature_info = {
        'time_varying_known_reals': training_dataset.reals,
        'time_varying_unknown_reals': training_dataset.reals,
        'categoricals': training_dataset.categoricals,
        'target': config.TARGET,
        'gold_price_col': config.GOLD_PRICE_COL,
    }
    
    with open(os.path.join(output_dir, 'feature_info.pkl'), 'wb') as f:
        pickle.dump(feature_info, f)
    
    print(f"\n✓ All artifacts saved to: {output_dir}/")
    print(f"  ✓ correlation_intelligence.pkl")
    print(f"  ✓ feature_importance.csv")
    print(f"  ✓ dataset_metadata.pkl")
    print(f"  ✓ feature_info.pkl")
    print(f"  ✓ config.pkl")
    print(f"  ✓ training_summary.csv")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main training pipeline
    """
    print("\n" + "="*70)
    print("🟡 GOLD MARKET CAUSAL INTELLIGENCE ENGINE")
    print("   Temporal Fusion Transformer Training")
    print("   Optimized for Your Master Dataset")
    print("="*70)
    print(f"\n⚙️  Configuration:")
    print(f"   Encoder: {Config.MAX_ENCODER_LENGTH} days | Prediction: {Config.MAX_PREDICTION_LENGTH} days")
    print(f"   Hidden: {Config.HIDDEN_SIZE} | Heads: {Config.ATTENTION_HEAD_SIZE} | Batch: {Config.BATCH_SIZE}")
    print(f"   Device: {Config.DEVICE.upper()}")
    print("="*70)
    
    # Initialize config
    config = Config()
    
    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    try:
        # Step 1: Load and prepare data
        df = load_and_prepare_data(config)
        
        # Step 2: Prepare features
        features = prepare_feature_lists(df, config)
        
        # Step 3: Create datasets
        training, validation, train_dataloader, val_dataloader = create_datasets(
            df, features, config
        )
        
        # Step 4: Build model
        tft, trainer = build_tft_model(training, config)
        
        # Step 5: Train
        best_model_path = train_model(tft, trainer, train_dataloader, val_dataloader)
        
        # Step 6: Extract feature importance
        interpretation, importance_df = extract_feature_importance(
            tft, val_dataloader, training, config.OUTPUT_DIR
        )
        
        # Step 7: Save artifacts
        save_training_artifacts(tft, training, config, config.OUTPUT_DIR, importance_df)
        
        print("\n" + "="*70)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("="*70)
        print(f"\n📁 Outputs saved:")
        print(f"   Models: {config.MODEL_DIR}/")
        print(f"   Artifacts: {config.OUTPUT_DIR}/")
        print(f"\n📊 Key Results:")
        print(f"   Best model: {os.path.basename(best_model_path)}")
        print(f"   Top feature: {importance_df.iloc[0]['feature']}")
        print(f"   Total features used: {len(training.reals + training.categoricals)}")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error above and retry.")


if __name__ == "__main__":
    main()