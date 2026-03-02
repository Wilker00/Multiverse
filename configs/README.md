# Configuration Examples

This directory contains example environment configurations for different deployment scenarios.

## Available Configurations

### [dev.env](dev.env)
**Development Environment** - Fast iteration, minimal safety overhead

- **Workers**: 2 (low parallelism for debugging)
- **Memory**: Small caches, ANN disabled, local storage
- **MCTS**: Reduced simulations (32), shallow depth (6)
- **Curriculum**: Disabled for manual control
- **Safety**: Relaxed thresholds
- **Use case**: Local development, quick experiments, debugging

### [staging.env](staging.env)
**Staging Environment** - Production-like evaluation

- **Workers**: 8 (moderate parallelism)
- **Memory**: Standard caches, FAISS enabled, shared storage
- **MCTS**: Full simulations (96), standard depth (12)
- **Curriculum**: Enabled with standard settings
- **Safety**: Full stack (planner + MCTS)
- **Use case**: Pre-production testing, integration tests, realistic evaluation

### [prod.env](prod.env)
**Production Environment** - Maximum robustness and performance

- **Workers**: 32 (high throughput)
- **Memory**: Large caches (50K queries), high ANN accuracy
- **MCTS**: Deep search (128 sims, depth 16)
- **Curriculum**: Conservative adjustments
- **Safety**: Stricter thresholds, full monitoring
- **Use case**: Production deployments, critical applications, large-scale training

## Usage

### Option 1: Symbolic Link (Recommended)

```bash
# Link the appropriate config for your environment
ln -s configs/dev.env .env        # Development
ln -s configs/staging.env .env    # Staging
ln -s configs/prod.env .env       # Production
```

### Option 2: Copy and Customize

```bash
# Copy to .env and customize
cp configs/dev.env .env
# Edit .env with your specific values
```

### Option 3: Source Directly

```bash
# Load config into current shell (for scripts)
set -a
source configs/dev.env
set +a

# Run your command
python tools/train_adt.py --verse labyrinth_world
```

### Option 4: Docker Compose Override

```bash
# For docker-compose deployments
docker-compose --env-file configs/prod.env up
```

## Customization

Each environment config can be customized by:

1. **Creating a local override**: `cp configs/prod.env .env.local` and edit
2. **Environment-specific secrets**: Store passwords/keys in secrets management
3. **Per-experiment tweaks**: Override specific vars: `MULTIVERSE_PARALLEL_NUM_WORKERS=16 python script.py`

## Configuration Precedence

When multiple config sources exist:

```
Command-line exports > .env.local > .env > configs/*.env > Code defaults
```

## Security Notes

### Development
- Uses placeholder passwords (change for shared dev environments)
- No external service credentials required

### Staging
- Uses `${VAR_NAME}` placeholders for secrets
- Set via: `export GRAFANA_STAGING_PASSWORD=<secret>`

### Production
- **NEVER commit real credentials to Git**
- Use secrets management: Kubernetes Secrets, AWS Secrets Manager, HashiCorp Vault
- Set `GF_SECURITY_ADMIN_PASSWORD`, `REDIS_PASSWORD`, etc. via environment
- Consider using `.env.local` (gitignored) for local overrides with real secrets

## Verification

Check your active configuration:

```bash
# Print all MULTIVERSE_* variables
env | grep MULTIVERSE_

# Verify specific setting
echo $MULTIVERSE_PARALLEL_NUM_WORKERS

# Test configuration loading
python -c "import os; print('Workers:', os.environ.get('MULTIVERSE_PARALLEL_NUM_WORKERS', 'NOT SET'))"
```

## Migration Guide

### From .env.example

1. Choose target environment: `configs/dev.env`, `staging.env`, or `prod.env`
2. Copy to `.env`: `cp configs/dev.env .env`
3. Uncomment and customize values as needed
4. For secrets, use environment variables or `.env.local`

### From Hardcoded Values

If you have configs hardcoded in Python:

**Before**:
```python
config = ParallelRolloutConfig(
    num_workers=4,
    max_worker_timeout_s=3600
)
```

**After**:
```python
# Uses environment variables automatically
config = ParallelRolloutConfig()  # Defaults from env

# Or explicit override:
config = ParallelRolloutConfig(
    num_workers=int(os.environ.get("MULTIVERSE_PARALLEL_NUM_WORKERS", "4"))
)
```

## Environment Comparison

| Setting | Development | Staging | Production |
|---------|-------------|---------|------------|
| **Workers** | 2 | 8 | 32 |
| **MCTS Sims** | 32 | 96 | 128 |
| **MCTS Depth** | 6 | 12 | 16 |
| **Query Cache** | 1K | 10K | 50K |
| **ANN Enabled** | No | Yes | Yes |
| **Curriculum** | Disabled | Enabled | Enabled |
| **Cache TTL** | 30s | 60s | 300s |
| **Worker Timeout** | 30min | 1hr | 2hr |
| **Priority** | Speed | Balance | Robustness |

## Troubleshooting

### Config Not Loading

```bash
# Check if .env exists
ls -la .env

# Check if symlink is correct
readlink .env

# Verify file permissions
chmod 644 .env
```

### Environment Not Taking Effect

```bash
# Ensure you're sourcing correctly
set -a && source .env && set +a

# Check if code is reading env vars
grep -r "os.environ.get" core/
```

### Docker Issues

```bash
# Verify Docker is using the env file
docker-compose config

# Pass env file explicitly
docker-compose --env-file configs/prod.env config
```

## See Also

- [Configuration Reference](../docs/CONFIGURATION.md) - Complete parameter documentation
- [Setup Guide](../docs/SETUP.md) - Installation instructions
- [.env.example](../.env.example) - Template with all available options

## Contributing

When adding new configuration options:

1. Update all three environment files (`dev.env`, `staging.env`, `prod.env`)
2. Document in [CONFIGURATION.md](../docs/CONFIGURATION.md)
3. Add to [.env.example](../.env.example) with comments
4. Update this README's comparison table
5. Test across all three environments
