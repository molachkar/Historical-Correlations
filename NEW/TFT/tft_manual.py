import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import torch
import warnings
import os
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')

class Config:
    DATA_PATH = "master_dataset.csv"
    OUTPUT_DIR = "tft_output"
    MODEL_DIR = "models"
    
    MAX_PREDICTION_LENGTH = 60
    MAX_ENCODER_LENGTH = 180
    
    BATCH_SIZE = 32
    MAX_EPOCHS = 100
    LEARNING_RATE = 0.0008
    HIDDEN_SIZE = 256
    ATTENTION_HEAD_SIZE = 8
    DROPOUT = 0.12
    GRADIENT_CLIP = 0.1
    PATIENCE = 12
    
    VALIDATION_CUTOFF_DATE = "2023-06-01"
    TARGET = "gold_return_future"
    GOLD_PRICE_COL = "Close_XAUUSD"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FeatureCategories:
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

def load_and_prepare_data(config):
    print("Loading data...")
    df = pd.read_csv(config.DATA_PATH)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    df['time_idx'] = (df['Date'] - df['Date'].min()).dt.days
    df['series_id'] = 'gold_macro'
    
    if 'Vol_Regime' in df.columns:
        df['Vol_Regime'] = df['Vol_Regime'].fillna('medium')
    
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_month'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    
    for col in FeatureCategories.ECONOMIC_INDICATORS:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill')
    
    df[config.TARGET] = (
        df[config.GOLD_PRICE_COL].shift(-config.MAX_PREDICTION_LENGTH) / 
        df[config.GOLD_PRICE_COL] - 1
    ) * 100
    
    df['gold_direction'] = (df[config.TARGET] > 0).astype(int)
    
    initial_len = len(df)
    df = df[df[config.TARGET].notna()].reset_index(drop=True)
    removed = initial_len - len(df)
    
    print(f"Target: {config.MAX_PREDICTION_LENGTH}-day ahead returns")
    print(f"Removed {removed} rows, final: {len(df)} rows")
    print(f"Target - mean: {df[config.TARGET].mean():.4f}%, std: {df[config.TARGET].std():.4f}%")
    
    return df

def prepare_feature_lists(df, config):
    print("Categorizing features...")
    
    available_cols = set(df.columns)
    
    time_varying_known_reals = [
        'time_idx', 'month', 'day_of_week', 'quarter', 
        'day_of_month', 'week_of_year'
    ]
    
    time_varying_known_reals.extend([
        col for col in FeatureCategories.ECONOMIC_INDICATORS 
        if col in available_cols
    ])
    
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
    
    time_varying_unknown_categoricals = [
        col for col in FeatureCategories.CATEGORICAL_FEATURES 
        if col in available_cols
    ]
    
    static_categoricals = ['series_id']
    
    total_features = len(time_varying_known_reals) + len(time_varying_unknown_reals) + len(time_varying_unknown_categoricals)
    print(f"Total features: {total_features} (known: {len(time_varying_known_reals)}, unknown: {len(time_varying_unknown_reals)}, categorical: {len(time_varying_unknown_categoricals)})")
    
    return {
        'time_varying_known_reals': time_varying_known_reals,
        'time_varying_unknown_reals': time_varying_unknown_reals,
        'time_varying_known_categoricals': [],
        'time_varying_unknown_categoricals': time_varying_unknown_categoricals,
        'static_categoricals': static_categoricals,
    }

def create_datasets(df, features, config):
    print("Creating datasets...")
    
    validation_cutoff = df[df['Date'] >= config.VALIDATION_CUTOFF_DATE]['time_idx'].min()
    
    train_df = df[df['time_idx'] < validation_cutoff].copy()
    val_df = df[df['time_idx'] >= validation_cutoff].copy()
    
    print(f"Train: {len(train_df)} samples, Valid: {len(val_df)} samples")
    
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
    
    validation = TimeSeriesDataSet.from_dataset(
        training, df, predict=True, stop_randomization=True
    )
    
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
    
    print(f"Batches - train: {len(train_dataloader)}, valid: {len(val_dataloader)}")
    
    return training, validation, train_dataloader, val_dataloader

def train_manual(tft, train_dataloader, val_dataloader, config):
    print("Training model...")
    print(f"Params: {sum(p.numel() for p in tft.parameters()):,}, Device: {config.DEVICE}")
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    tft = tft.to(config.DEVICE)
    optimizer = torch.optim.Adam(tft.parameters(), lr=config.LEARNING_RATE)
    loss_fn = QuantileLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = None
    
    for epoch in range(config.MAX_EPOCHS):
        tft.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
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
            loss = loss_fn(predictions, y[0])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tft.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        tft.eval()
        val_losses = []
        
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
                loss = loss_fn(predictions, y[0])
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        print(f"Epoch {epoch+1}/{config.MAX_EPOCHS} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(config.MODEL_DIR, f"tft-epoch{epoch+1:02d}-{val_loss:.4f}.pth")
            torch.save(tft.state_dict(), best_model_path)
            print(f"  Saved best model: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Training complete - Best val loss: {best_val_loss:.4f}")
    
    if best_model_path:
        tft.load_state_dict(torch.load(best_model_path))
    
    return tft, best_model_path, best_val_loss

def extract_feature_importance(tft, val_dataloader, output_dir):
    print("Extracting feature importance...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    tft.eval()
    
    batch = next(iter(val_dataloader))
    x, y = batch
    
    for key in x:
        if isinstance(x[key], torch.Tensor):
            x[key] = x[key].to(tft.device)
    
    with torch.no_grad():
        interpretation = tft.interpret_output(x, reduction="sum")
    
    encoder_vars = interpretation.get('encoder_variables', {})
    decoder_vars = interpretation.get('decoder_variables', {})
    static_vars = interpretation.get('static_variables', {})
    
    all_importance = {}
    all_importance.update(encoder_vars)
    all_importance.update(decoder_vars)
    all_importance.update(static_vars)
    
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v, 'type': 'encoder' if k in encoder_vars else 'decoder' if k in decoder_vars else 'static'}
        for k, v in all_importance.items()
    ]).sort_values('importance', ascending=False)
    
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    with open(os.path.join(output_dir, 'interpretation.pkl'), 'wb') as f:
        pickle.dump(interpretation, f)
    
    print("Top 15 features:")
    for idx, row in importance_df.head(15).iterrows():
        print(f"  {row['feature']:<35} {row['importance']:>8.4f}")
    
    return interpretation, importance_df

def save_artifacts(tft, training_dataset, config, output_dir, importance_df, best_val_loss):
    print("Saving artifacts...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'dataset_metadata.pkl'), 'wb') as f:
        pickle.dump({
            'parameters': training_dataset.get_parameters(),
            'scalers': training_dataset.scalers,
            'categorical_encoders': training_dataset.categorical_encoders,
        }, f)
    
    with open(os.path.join(output_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)
    
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
        'best_val_loss': best_val_loss,
        'top_feature': importance_df.iloc[0]['feature'] if len(importance_df) > 0 else 'N/A',
        'top_importance': importance_df.iloc[0]['importance'] if len(importance_df) > 0 else 0,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, 'training_summary.csv'), index=False)
    
    print(f"Saved to {output_dir}/")

def main():
    print("TFT Gold Prediction - Manual Training")
    print(f"Config: Encoder={Config.MAX_ENCODER_LENGTH}, Predict={Config.MAX_PREDICTION_LENGTH}, Hidden={Config.HIDDEN_SIZE}")
    print(f"Device: {Config.DEVICE}\n")
    
    config = Config()
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    try:
        df = load_and_prepare_data(config)
        print("")
        
        features = prepare_feature_lists(df, config)
        print("")
        
        training, validation, train_dataloader, val_dataloader = create_datasets(df, features, config)
        print("")
        
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=config.LEARNING_RATE,
            hidden_size=config.HIDDEN_SIZE,
            attention_head_size=config.ATTENTION_HEAD_SIZE,
            dropout=config.DROPOUT,
            hidden_continuous_size=config.HIDDEN_SIZE // 2,
            output_size=7,
            loss=QuantileLoss(),
        )
        
        tft, best_model_path, best_val_loss = train_manual(tft, train_dataloader, val_dataloader, config)
        print("")
        
        interpretation, importance_df = extract_feature_importance(tft, val_dataloader, config.OUTPUT_DIR)
        print("")
        
        save_artifacts(tft, training, config, config.OUTPUT_DIR, importance_df, best_val_loss)
        print("")
        
        print("Complete")
        print(f"Model: {os.path.basename(best_model_path) if best_model_path else 'N/A'}")
        print(f"Top feature: {importance_df.iloc[0]['feature']}")
        print(f"Best val loss: {best_val_loss:.4f}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()