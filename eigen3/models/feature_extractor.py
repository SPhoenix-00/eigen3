"""Feature extraction with CNN and Bidirectional LSTM (JAX/Flax implementation)

Converts PyTorch FeatureExtractor to JAX/Flax with:
- 1D CNN across features
- Bidirectional LSTM across time
- Column chunking for memory efficiency
- Gradient checkpointing (remat) support
"""

from typing import Sequence
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import checkpoint as remat
import chex


class BiLSTM(nn.Module):
    """Bidirectional LSTM for temporal processing

    Implements a stacked bidirectional LSTM that processes sequences
    and returns the concatenated forward/backward hidden states.
    """
    hidden_size: int
    num_layers: int = 2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: chex.Array, train: bool = True) -> chex.Array:
        """Apply bidirectional LSTM

        Args:
            x: Input tensor [batch, time_steps, features]
            train: Whether in training mode (for dropout)

        Returns:
            Output tensor [batch, time_steps, 2*hidden_size]
        """
        batch_size, time_steps, input_size = x.shape

        # Initialize carry states for forward and backward LSTMs
        # Flax LSTM uses (c, h) as carry
        carry_shape = (batch_size, self.hidden_size)

        # Process with stacked bidirectional LSTM layers
        hidden = x

        for layer_idx in range(self.num_layers):
            # Forward LSTM
            lstm_cell_fwd = nn.LSTMCell(name=f'lstm_fwd_layer{layer_idx}')

            def fwd_scan_fn(carry, x_t):
                carry, y = lstm_cell_fwd(carry, x_t)
                return carry, y

            # Initialize forward carry
            carry_fwd = lstm_cell_fwd.initialize_carry(
                jax.random.PRNGKey(0),
                (batch_size,),
                self.hidden_size
            )

            # Scan forward through time
            _, hidden_fwd = jax.lax.scan(fwd_scan_fn, carry_fwd, jnp.transpose(hidden, (1, 0, 2)))
            hidden_fwd = jnp.transpose(hidden_fwd, (1, 0, 2))  # [batch, time, hidden]

            # Backward LSTM
            lstm_cell_bwd = nn.LSTMCell(name=f'lstm_bwd_layer{layer_idx}')

            def bwd_scan_fn(carry, x_t):
                carry, y = lstm_cell_bwd(carry, x_t)
                return carry, y

            # Initialize backward carry
            carry_bwd = lstm_cell_bwd.initialize_carry(
                jax.random.PRNGKey(0),
                (batch_size,),
                self.hidden_size
            )

            # Scan backward through time (reverse the sequence)
            _, hidden_bwd = jax.lax.scan(bwd_scan_fn, carry_bwd, jnp.transpose(hidden[:, ::-1], (1, 0, 2)))
            hidden_bwd = jnp.transpose(hidden_bwd[::-1], (1, 0, 2))  # [batch, time, hidden], unreversed

            # Concatenate forward and backward
            hidden = jnp.concatenate([hidden_fwd, hidden_bwd], axis=-1)  # [batch, time, 2*hidden]

            # Apply dropout between layers (except last layer)
            if layer_idx < self.num_layers - 1 and self.dropout_rate > 0:
                hidden = nn.Dropout(self.dropout_rate, deterministic=not train)(hidden)

        return hidden


class FeatureExtractor(nn.Module):
    """Extract features from multi-column time-series data (Eigen2-aligned).

    Architecture:
    1. Instance normalization first (per column, across time; scale-invariant)
    2. 1D CNN across features (per column)
    3. Bidirectional LSTM across time
    4. Average pooling of last 3 timesteps
    5. Column chunking for memory efficiency

    Synced with Eigen2: Instance Norm before CNN; default 117 columns.
    """
    num_columns: int = 117  # Eigen2 TOTAL_COLUMNS (skinny)
    num_features: int = 5  # [close, RSI, MACD_signal, TRIX, diff20DMA]
    cnn_filters: int = 32
    cnn_kernel_size: int = 3
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True
    column_chunk_size: int = 64  # Process columns in chunks
    use_remat: bool = True  # Gradient checkpointing

    def setup(self):
        """Initialize submodules"""
        # CNN layer
        self.conv1 = nn.Conv(
            features=self.cnn_filters,
            kernel_size=(self.cnn_kernel_size,),
            padding='SAME',
            name='conv1'
        )
        self.bn1 = nn.BatchNorm(name='bn1')

        # LSTM
        self.lstm = BiLSTM(
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout_rate=0.1,
            name='bilstm'
        )

        # Output size
        self.lstm_output_size = self.lstm_hidden_size * (2 if self.lstm_bidirectional else 1)

    def _cnn_block(self, x: chex.Array, train: bool) -> chex.Array:
        """CNN processing block for gradient checkpointing

        Args:
            x: Input tensor [batch*chunk_size, num_features, time_steps]
            train: Whether in training mode

        Returns:
            Features [batch*chunk_size, time_steps, cnn_filters]
        """
        # Conv1D across features
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = nn.relu(x)
        return x

    def _lstm_block(self, x: chex.Array, train: bool) -> chex.Array:
        """LSTM processing block for gradient checkpointing

        Args:
            x: Input tensor [batch*chunk_size, time_steps, cnn_filters]
            train: Whether in training mode

        Returns:
            LSTM output [batch*chunk_size, time_steps, lstm_output_size]
        """
        # Replace NaN with 0 (matching PyTorch nan_to_num)
        x = jnp.nan_to_num(x, nan=0.0)

        # Apply bidirectional LSTM
        lstm_out = self.lstm(x, train=train)
        return lstm_out

    def _process_chunk(
        self,
        x_chunk: chex.Array,
        batch_size: int,
        chunk_size: int,
        train: bool
    ) -> chex.Array:
        """Process a single chunk of columns

        Args:
            x_chunk: Chunk tensor [batch, time_steps, chunk_size, num_features]
            batch_size: Batch size
            chunk_size: Number of columns in this chunk
            train: Whether in training mode

        Returns:
            Features [batch, chunk_size, lstm_output_size]
        """
        time_steps = x_chunk.shape[1]

        # Reshape for CNN: [batch, chunk_size, num_features, time_steps]
        x_chunk = jnp.transpose(x_chunk, (0, 2, 3, 1))

        # Flatten batch and chunk dimensions: [batch*chunk_size, num_features, time_steps]
        x_chunk = x_chunk.reshape(batch_size * chunk_size, self.num_features, time_steps)

        # Eigen2: Instance normalization first (per column, across time; affine=False)
        # Makes model scale-invariant (raw nominal data)
        mean = jnp.mean(x_chunk, axis=2, keepdims=True)
        var = jnp.var(x_chunk, axis=2, keepdims=True) + 1e-5
        x_chunk = (x_chunk - mean) * jax.lax.rsqrt(var)

        # CNN across features (with optional gradient checkpointing)
        if train and self.use_remat:
            x_chunk = remat(lambda x: self._cnn_block(x, train))(x_chunk)
        else:
            x_chunk = self._cnn_block(x_chunk, train)

        # Reshape for LSTM: [batch*chunk_size, time_steps, cnn_filters]
        x_chunk = jnp.transpose(x_chunk, (0, 2, 1))

        # LSTM across time (with optional gradient checkpointing)
        if train and self.use_remat:
            lstm_out = remat(lambda x: self._lstm_block(x, train))(x_chunk)
        else:
            lstm_out = self._lstm_block(x_chunk, train)

        # Average of last 3 timesteps: [batch*chunk_size, lstm_output_size]
        x_chunk = jnp.mean(lstm_out[:, -3:, :], axis=1)

        # Reshape back: [batch, chunk_size, lstm_output_size]
        x_chunk = x_chunk.reshape(batch_size, chunk_size, self.lstm_output_size)

        return x_chunk

    @nn.compact
    def __call__(self, x: chex.Array, train: bool = True) -> chex.Array:
        """Forward pass with column chunking and gradient checkpointing

        Column chunking prevents OOM by processing columns in smaller batches.
        This keeps the effective LSTM batch size manageable.

        Args:
            x: Input tensor [batch, time_steps, num_columns, num_features]
               (Note: PyTorch uses [batch, context_days, num_columns, num_features])
            train: Whether in training mode (for BatchNorm and Dropout)

        Returns:
            Features tensor [batch, num_columns, lstm_output_size]
        """
        batch_size = x.shape[0]
        time_steps = x.shape[1]

        # Validate input shape
        assert x.shape[2] == self.num_columns, f"Expected {self.num_columns} columns, got {x.shape[2]}"
        assert x.shape[3] == self.num_features, f"Expected {self.num_features} features, got {x.shape[3]}"

        # Process columns in chunks
        num_chunks = (self.num_columns + self.column_chunk_size - 1) // self.column_chunk_size
        chunk_outputs = []

        for chunk_idx in range(num_chunks):
            # Get column indices for this chunk
            start_col = chunk_idx * self.column_chunk_size
            end_col = min(start_col + self.column_chunk_size, self.num_columns)
            chunk_size = end_col - start_col

            # Extract chunk: [batch, time_steps, chunk_size, num_features]
            x_chunk = x[:, :, start_col:end_col, :]

            # Process chunk
            chunk_output = self._process_chunk(x_chunk, batch_size, chunk_size, train)

            # Store chunk output
            chunk_outputs.append(chunk_output)

        # Concatenate all chunks: [batch, num_columns, lstm_output_size]
        output = jnp.concatenate(chunk_outputs, axis=1)

        return output


def test_feature_extractor():
    """Test the FeatureExtractor implementation"""
    import jax.random as random

    # Create model (Eigen2: 117 columns, 151 context days)
    model = FeatureExtractor(
        num_columns=117,
        num_features=5,
        cnn_filters=32,
        lstm_hidden_size=128,
        lstm_num_layers=2,
        column_chunk_size=64,
        use_remat=False  # Disable for testing
    )

    # Create dummy input
    key = random.PRNGKey(0)
    batch_size = 2
    context_days = 151
    x = random.normal(key, (batch_size, context_days, 117, 5))

    # Initialize parameters
    params = model.init(key, x, train=False)

    # Forward pass
    output = model.apply(params, x, train=False)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 117, 256)")

    assert output.shape == (batch_size, 117, 256), f"Output shape mismatch: {output.shape}"
    print("✓ FeatureExtractor test passed!")

    # Test with gradient checkpointing
    model_remat = FeatureExtractor(
        num_columns=117,
        num_features=5,
        use_remat=True
    )
    params_remat = model_remat.init(key, x, train=False)
    output_remat = model_remat.apply(params_remat, x, train=True)

    print(f"✓ Gradient checkpointing test passed! Output shape: {output_remat.shape}")

    return model, params, output


if __name__ == "__main__":
    test_feature_extractor()
