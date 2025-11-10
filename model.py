import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class FinancialTimeSeriesDataset(Dataset):
    """Enhanced Dataset with volatility regime awareness"""

    def __init__(self, features, targets, prices, sequence_length=60, volatility_window=20):
        self.features = features
        self.targets = targets
        self.prices = prices
        self.sequence_length = sequence_length
        
        # Calculate volatility regime for each sample
        self.volatility_regimes = self._calculate_volatility_regimes(volatility_window)
        self.valid_indices = list(range(sequence_length, len(features)))

    def _calculate_volatility_regimes(self, window):
        """Calculate rolling volatility to identify regime changes"""
        returns = np.diff(self.prices) / self.prices[:-1]
        returns = np.concatenate([[0], returns])  # Prepend 0 for alignment
        
        volatilities = []
        for i in range(len(returns)):
            if i < window:
                vol = np.std(returns[:i+1]) if i > 0 else 0
            else:
                vol = np.std(returns[i-window:i])
            volatilities.append(vol)
        
        return np.array(volatilities)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        # Extract sequence
        feature_sequence = self.features[actual_idx - self.sequence_length:actual_idx]
        
        # Extract volatility regime
        volatility_regime = self.volatility_regimes[actual_idx]
        
        # Extract target
        target = self.targets[actual_idx]
        
        return (torch.FloatTensor(feature_sequence), 
                torch.FloatTensor([volatility_regime]),
                torch.FloatTensor(target))

class VolatilityAwareAttention(nn.Module):
    """Attention mechanism that adapts to volatility regimes"""
    
    def __init__(self, hidden_size, num_heads=8):
        super(VolatilityAwareAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
        # Volatility-aware temperature scaling
        self.temp_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, volatility):
        batch_size, seq_len, hidden_size = x.size()
        
        # Calculate temperature based on volatility
        temperature = self.temp_mlp(volatility) + 1.0  # Ensure temperature >= 1
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled attention with volatility-aware temperature
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (np.sqrt(self.head_dim) * temperature.unsqueeze(-1).unsqueeze(-1))
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.out(context)
        
        # Weighted pooling (give more weight to recent observations)
        position_weights = torch.linspace(0.5, 1.0, seq_len).to(x.device)
        position_weights = position_weights.view(1, seq_len, 1).expand_as(output)
        pooled_output = (output * position_weights).sum(dim=1) / position_weights.sum(dim=1)
        
        return pooled_output, attention_weights.mean(dim=1)

class TemporalConvBlock(nn.Module):
    """Temporal convolutional block for capturing different time scales"""
    
    def __init__(self, in_channels, out_channels):
        super(TemporalConvBlock, self).__init__()
        
        # Ensure channels divide evenly
        channels_per_scale = out_channels // 3
        remaining_channels = out_channels - (channels_per_scale * 2)
        
        # Multi-scale convolutions
        self.conv_small = nn.Conv1d(in_channels, channels_per_scale, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(in_channels, channels_per_scale, kernel_size=5, padding=2)
        self.conv_large = nn.Conv1d(in_channels, remaining_channels, kernel_size=7, padding=3)
        
        # Calculate actual output channels
        self.actual_out_channels = channels_per_scale * 2 + remaining_channels
        
        self.bn = nn.BatchNorm1d(self.actual_out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Multi-scale feature extraction
        small = self.conv_small(x)
        medium = self.conv_medium(x)
        large = self.conv_large(x)
        
        # Concatenate multi-scale features
        out = torch.cat([small, medium, large], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        return out

class ImprovedHybridFinancialModel(nn.Module):
    """Improved model with better volatility capture and sharper predictions"""
    
    def __init__(self, input_size, hidden_size=256, num_layers=3, dropout=0.3, num_heads=8):
        super(ImprovedHybridFinancialModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(0.1)
        
        # Branch 1: Bidirectional LSTM with volatility-aware attention
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.volatility_attention = VolatilityAwareAttention(hidden_size * 2, num_heads)
        
        # Branch 2: Multi-scale temporal CNN
        self.temporal_conv1 = TemporalConvBlock(input_size, 128)
        self.temporal_conv2 = TemporalConvBlock(self.temporal_conv1.actual_out_channels, 256)
        self.temporal_pool = nn.AdaptiveMaxPool1d(1)  # Use max pooling to capture peaks
        
        # Branch 3: GRU for recent pattern focus
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout, bidirectional=False)
        
        # Volatility regime encoder
        self.volatility_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # Feature fusion with gating mechanism
        # lstm: hidden_size * 2, cnn: temporal_conv2.actual_out_channels, gru: hidden_size, vol: 64
        fusion_input_size = hidden_size * 2 + self.temporal_conv2.actual_out_channels + hidden_size + 64
        self.feature_gate = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_input_size),
            nn.Sigmoid()
        )
        
        self.feature_combiner = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual processing
        self.residual1 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.residual2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Separate prediction heads with increased capacity
        self.prediction_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_size // 2, 1)
            ) for _ in range(5)
        ])
        
        # Volatility prediction head (predicts future volatility)
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 5)
        )
    
    def forward(self, x, volatility):
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x_proj = self.input_projection(x)
        x_proj = self.input_dropout(x_proj)
        
        # Branch 1: LSTM with volatility-aware attention
        lstm_out, _ = self.lstm(x_proj)
        lstm_context, _ = self.volatility_attention(lstm_out, volatility)
        
        # Branch 2: Multi-scale temporal CNN (captures sharp changes better)
        x_cnn = x.transpose(1, 2)
        cnn_out = self.temporal_conv1(x_cnn)
        cnn_out = self.temporal_conv2(cnn_out)
        cnn_out = self.temporal_pool(cnn_out).squeeze(-1)
        
        # Branch 3: GRU for recent patterns
        gru_out, _ = self.gru(x_proj)
        gru_context = gru_out[:, -1, :]  # Take last timestep
        
        # Volatility regime encoding
        vol_encoded = self.volatility_encoder(volatility)
        
        # Combine all features with gating
        combined = torch.cat([lstm_context, cnn_out, gru_context, vol_encoded], dim=1)
        gate = self.feature_gate(combined)
        gated_features = combined * gate
        
        features = self.feature_combiner(gated_features)
        
        # Residual processing
        features = features + self.residual1(features)
        features = features + self.residual2(features)
        
        # Generate predictions
        predictions = []
        for layer in self.prediction_layers:
            pred = layer(features)
            predictions.append(pred)
        
        predictions = torch.cat(predictions, dim=1)
        
        # Predict future volatility
        pred_volatility = torch.nn.functional.softplus(self.volatility_head(features))
        
        return predictions, pred_volatility

def create_enhanced_features(df):
    """Create features focusing on momentum and volatility"""
    
    print("Available columns:", df.columns.tolist())
    
    # Enhanced momentum features
    for period in [3, 5, 10, 20]:
        df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        df[f'momentum_accel_{period}'] = df[f'momentum_{period}'].diff()
    
    # High-low range features (only if High/Low exist)
    if 'High' in df.columns and 'Low' in df.columns:
        df['hl_range'] = (df['High'] - df['Low']) / df['Close']
        df['hl_range_ma5'] = df['hl_range'].rolling(5).mean()
        df['hl_range_ma20'] = df['hl_range'].rolling(20).mean()
    else:
        # Use close price range as proxy
        for period in [5, 20]:
            df[f'close_range_{period}'] = (df['Close'].rolling(period).max() - 
                                           df['Close'].rolling(period).min()) / df['Close']
    
    # Price position in recent range
    for period in [5, 10, 20]:
        price_min = df['Close'].rolling(period).min()
        price_max = df['Close'].rolling(period).max()
        df[f'price_position_{period}'] = (df['Close'] - price_min) / (price_max - price_min + 1e-8)
    
    # Volatility features (multiple windows)
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['Return_1d'].rolling(period).std()
        df[f'volatility_change_{period}'] = df[f'volatility_{period}'].diff()
    
    # Volume features (only if Volume exists)
    if 'Volume' in df.columns:
        df['volume_ma5'] = df['Volume'].rolling(5).mean()
        df['volume_ma20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_ma5'] + 1e-8)
        df['volume_momentum'] = df['volume_ma5'] / (df['volume_ma20'] + 1e-8)
    
    # Price acceleration
    df['price_velocity'] = df['Close'].diff()
    df['price_acceleration'] = df['price_velocity'].diff()
    
    # Moving average convergence/divergence
    df['ma5'] = df['Close'].rolling(5).mean()
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma_diff'] = (df['ma5'] - df['ma20']) / df['Close']
    df['ma_diff_change'] = df['ma_diff'].diff()
    
    # Recent lagged features (more emphasis on recent past)
    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = df['Return_1d'].shift(lag)
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
    
    # Price rate of change
    for period in [3, 7, 14]:
        df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
    
    # Forward fill and drop NaN
    df = df.fillna(method='ffill').dropna()
    
    print(f"Features after engineering: {df.shape[1]} columns")
    
    return df

def load_and_prepare_enhanced_data(csv_path, sequence_length=60, test_size=0.2):
    """Load and prepare data with volatility awareness"""
    
    print("Loading and preparing data...")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded data shape: {df.shape}")
    
    # Sort by date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    elif 'time_idx' in df.columns:
        df = df.sort_values('time_idx').reset_index(drop=True)
    
    # Create enhanced features
    df = create_enhanced_features(df)
    print(f"Enhanced data shape: {df.shape}")
    
    # Select features
    exclude_cols = ['Date', 'time_idx', 'group_id', 'target', 'year', 'month',
                   'day_of_month', 'day_of_week', 'is_month_end']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    features = df[feature_cols].values
    close_prices = df['Close'].values
    
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    
    # Create targets (percentage changes)
    targets = []
    for i in range(len(close_prices) - 5):
        current_price = close_prices[i]
        future_prices = close_prices[i+1:i+6]
        percentage_changes = (future_prices - current_price) / current_price
        targets.append(percentage_changes)
    
    targets = np.array(targets)
    features = features[:-5]
    prices = close_prices[:-5]
    
    # Use RobustScaler
    feature_scaler = RobustScaler()
    target_scaler = StandardScaler()
    
    features_normalized = feature_scaler.fit_transform(features)
    targets_normalized = target_scaler.fit_transform(targets)
    
    # Chronological split
    split_idx = int(len(features_normalized) * (1 - test_size))
    
    train_features = features_normalized[:split_idx]
    train_targets = targets_normalized[:split_idx]
    train_prices = prices[:split_idx]
    
    test_features = features_normalized[split_idx:]
    test_targets = targets_normalized[split_idx:]
    test_prices = prices[split_idx:]
    
    print(f"Train samples: {len(train_features)}")
    print(f"Test samples: {len(test_features)}")
    
    # Create datasets with volatility awareness
    train_dataset = FinancialTimeSeriesDataset(train_features, train_targets, train_prices, sequence_length)
    test_dataset = FinancialTimeSeriesDataset(test_features, test_targets, test_prices, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, feature_scaler, target_scaler, feature_cols, test_prices

class CustomLoss(nn.Module):
    """Custom loss that penalizes direction errors and smoothness"""
    
    def __init__(self, direction_weight=2.0, smoothness_penalty=0.1):
        super(CustomLoss, self).__init__()
        self.direction_weight = direction_weight
        self.smoothness_penalty = smoothness_penalty
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        # Base MSE loss
        mse_loss = self.mse(predictions, targets)
        
        # Direction loss (penalize wrong direction more heavily)
        pred_sign = torch.sign(predictions)
        target_sign = torch.sign(targets)
        direction_mismatch = (pred_sign != target_sign).float()
        direction_loss = direction_mismatch.mean() * self.direction_weight
        
        # Smoothness penalty (encourage sharper predictions)
        # Penalize predictions that are too close to zero (too smooth)
        smoothness_loss = torch.exp(-torch.abs(predictions)).mean() * self.smoothness_penalty
        
        total_loss = mse_loss + direction_loss + smoothness_loss
        
        return total_loss

class EarlyStopping:
    """Early stopping with model checkpoint"""
    
    def __init__(self, patience=15, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict().copy()
        else:
            self.counter += 1
        
        return self.counter >= self.patience

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.0005):
    """Train with custom loss and better optimization"""
    
    print("\nStarting training...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Use custom loss
    criterion = CustomLoss(direction_weight=2.5, smoothness_penalty=0.05)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    early_stopping = EarlyStopping(patience=20)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for features, volatility, targets in train_loader:
            features = features.to(device)
            volatility = volatility.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            predictions, pred_vol = model(features, volatility)
            
            loss = criterion(predictions, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for features, volatility, targets in val_loader:
                features = features.to(device)
                volatility = volatility.to(device)
                targets = targets.to(device)
                
                predictions, _ = model(features, volatility)
                loss = criterion(predictions, targets)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Step scheduler
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, LR: {lr:.8f}')
        
        if early_stopping(avg_val_loss, model):
            print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(early_stopping.best_model)
            break
    
    print("Training completed!")
    return train_losses, val_losses

def evaluate_model(model, test_loader, target_scaler, test_prices):
    """Evaluate with detailed metrics"""
    
    print("\nEvaluating model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_pred_vol = []
    all_actuals = []
    all_base_prices = []
    
    with torch.no_grad():
        for i, (features, volatility, targets) in enumerate(test_loader):
            features = features.to(device)
            volatility = volatility.to(device)
            targets = targets.to(device)
            
            predictions, pred_vol = model(features, volatility)
            
            batch_size = features.size(0)
            start_idx = i * test_loader.batch_size
            end_idx = min(start_idx + batch_size, len(test_prices) - 60)
            
            if start_idx < len(test_prices) - 60:
                batch_base_prices = test_prices[start_idx + 60:end_idx + 60]
                if len(batch_base_prices) < batch_size:
                    batch_base_prices = np.pad(batch_base_prices,
                                             (0, batch_size - len(batch_base_prices)),
                                             'constant', constant_values=batch_base_prices[-1] if len(batch_base_prices) > 0 else 0)
                batch_base_prices = batch_base_prices[:batch_size]
            else:
                batch_base_prices = np.full(batch_size, test_prices[-1])
            
            all_predictions.append(predictions.cpu().numpy())
            all_pred_vol.append(pred_vol.cpu().numpy())
            all_actuals.append(targets.cpu().numpy())
            all_base_prices.append(batch_base_prices)
    
    predictions = np.concatenate(all_predictions, axis=0)
    pred_volatilities = np.concatenate(all_pred_vol, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    base_prices = np.concatenate(all_base_prices, axis=0)
    
    # Inverse transform
    predictions_normalized = target_scaler.inverse_transform(predictions)
    actuals_normalized = target_scaler.inverse_transform(actuals)
    
    # Convert to prices
    predictions_prices = []
    actuals_prices = []
    
    for i in range(len(predictions_normalized)):
        base_price = base_prices[i]
        pred_prices = base_price * (1 + predictions_normalized[i])
        actual_prices = base_price * (1 + actuals_normalized[i])
        predictions_prices.append(pred_prices)
        actuals_prices.append(actual_prices)
    
    predictions_prices = np.array(predictions_prices)
    actuals_prices = np.array(actuals_prices)
    
    # Metrics
    mse = np.mean((predictions_prices - actuals_prices) ** 2)
    mae = np.mean(np.abs(predictions_prices - actuals_prices))
    mape = np.mean(np.abs((predictions_prices - actuals_prices) / actuals_prices)) * 100
    
    # Direction accuracy
    pred_directions = np.sign(predictions_normalized)
    actual_directions = np.sign(actuals_normalized)
    direction_accuracy = np.mean(pred_directions == actual_directions) * 100
    
    # Sharpness metric (how much predictions deviate from baseline)
    prediction_variance = np.var(predictions_normalized)
    actual_variance = np.var(actuals_normalized)
    sharpness_ratio = prediction_variance / actual_variance
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test MAPE: {mape:.2f}%")
    print(f"Direction Accuracy: {direction_accuracy:.2f}%")
    print(f"Sharpness Ratio: {sharpness_ratio:.4f} (higher is better, closer to 1 is ideal)")
    
    print("\nExample predictions (first 5 samples):")
    for i in range(min(5, len(predictions_prices))):
        print(f"\nSample {i+1} (Base: ${base_prices[i]:.2f}):")
        print(f"  Predicted: {predictions_prices[i]}")
        print(f"  Actual:    {actuals_prices[i]}")
        print(f"  Error:     {predictions_prices[i] - actuals_prices[i]}")
    
    return predictions_prices, actuals_prices, pred_volatilities

def plot_results(predictions, actuals, train_losses, val_losses, volatilities):
    """Enhanced plotting"""
    
    plt.figure(figsize=(20, 15))
    
    # Training progress
    plt.subplot(3, 3, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Predictions vs Actuals for each horizon
    for horizon in range(5):
        plt.subplot(3, 3, horizon + 2)
        plt.scatter(actuals[:, horizon], predictions[:, horizon], alpha=0.5, s=15)
        
        min_val = min(actuals[:, horizon].min(), predictions[:, horizon].min())
        max_val = max(actuals[:, horizon].max(), predictions[:, horizon].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8)
        
        r2 = 1 - np.sum((actuals[:, horizon] - predictions[:, horizon])**2) / \
             np.sum((actuals[:, horizon] - np.mean(actuals[:, horizon]))**2)
        
        plt.title(f'Predictions vs Actual (t+{horizon+1}) - R²={r2:.3f}')
        plt.xlabel(f'Actual Price (t+{horizon+1})')
        plt.ylabel(f'Predicted Price (t+{horizon+1})')
        plt.grid(True, alpha=0.3)
    
    # Time series with volatility bands
    plt.subplot(3, 3, 7)
    sample_size = min(200, len(predictions))
    indices = range(sample_size)
    
    plt.plot(indices, actuals[:sample_size, 0], label='Actual (t+1)', alpha=0.8, linewidth=2)
    plt.plot(indices, predictions[:sample_size, 0], label='Predicted (t+1)', alpha=0.8, linewidth=2)
    
    upper_bound = predictions[:sample_size, 0] + volatilities[:sample_size, 0]
    lower_bound = predictions[:sample_size, 0] - volatilities[:sample_size, 0]
    plt.fill_between(indices, lower_bound, upper_bound, alpha=0.2, label='Volatility Band')
    
    plt.title('Time Series Comparison with Volatility')
    plt.xlabel('Sample Index')
    plt.ylabel('Price (t+1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    plt.subplot(3, 3, 8)
    errors = predictions[:, 0] - actuals[:, 0]
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
    plt.axvline(0, color='black', linestyle='-', alpha=0.5)
    plt.title('Prediction Error Distribution (t+1)')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MAE by horizon
    plt.subplot(3, 3, 9)
    horizons = ['t+1', 't+2', 't+3', 't+4', 't+5']
    maes = [np.mean(np.abs(predictions[:, i] - actuals[:, i])) for i in range(5)]
    
    x = np.arange(len(horizons))
    plt.bar(x, maes, alpha=0.7, color='steelblue')
    plt.title('MAE by Prediction Horizon')
    plt.xlabel('Prediction Horizon')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(x, horizons)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    
    CSV_PATH = "datacleaned.csv"
    SEQUENCE_LENGTH = 60
    HIDDEN_SIZE = 256
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0005
    
    try:
        # Load data
        train_loader, test_loader, feature_scaler, target_scaler, feature_cols, test_prices = load_and_prepare_enhanced_data(
            CSV_PATH, SEQUENCE_LENGTH
        )
        
        input_size = len(feature_cols)
        print(f"\nModel input size: {input_size}")
        
        # Create model
        model = ImprovedHybridFinancialModel(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=3,
            dropout=0.3,
            num_heads=8
        )
        
        print(f"\nModel architecture:")
        print(model)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Create validation split
        train_size = len(train_loader.dataset)
        val_size = int(0.15 * train_size)
        train_size = train_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_loader.dataset, [train_size, val_size]
        )
        
        train_loader_new = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train
        train_losses, val_losses = train_model(
            model, train_loader_new, val_loader, NUM_EPOCHS, LEARNING_RATE
        )
        
        # Evaluate
        predictions, actuals, volatilities = evaluate_model(
            model, test_loader, target_scaler, test_prices
        )
        
        # Plot
        plot_results(predictions, actuals, train_losses, val_losses, volatilities)
        
        print("\nImproved pipeline completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{CSV_PATH}'")
        print("Please update CSV_PATH with the correct path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
