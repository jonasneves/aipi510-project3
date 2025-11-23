# Cloudflare Tunnel Hosting

Host the API and frontend on GitHub Actions with Cloudflare tunnels.

## Prerequisites

1. Cloudflare account with domain
2. AWS S3 with trained model (run ML pipeline first)
3. GitHub secrets configured

## Setup

### 1. Create Cloudflare Tunnel

```bash
# Install cloudflared
brew install cloudflare/cloudflare/cloudflared  # macOS

# Login and create tunnel
cloudflared tunnel login
cloudflared tunnel create salary-predictor

# Note the tunnel ID
```

### 2. Configure Tunnel Routes

In Cloudflare Zero Trust dashboard (or via CLI):

```bash
# Route API
cloudflared tunnel route dns salary-predictor api-salary.neevs.io

# Route Frontend
cloudflared tunnel route dns salary-predictor app-salary.neevs.io
```

### 3. Create Tunnel Config

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: <TUNNEL_ID>
credentials-file: /root/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: api-salary.neevs.io
    service: http://localhost:8000
  - hostname: app-salary.neevs.io
    service: http://localhost:8501
  - service: http_status:404
```

### 4. Get Tunnel Token

```bash
cloudflared tunnel token <TUNNEL_ID>
```

### 5. Add GitHub Secrets

| Secret | Description |
|--------|-------------|
| `CLOUDFLARE_TUNNEL_TOKEN` | From step 4 |
| `AWS_ROLE_ARN` | IAM role for S3 access |

## Architecture

```
Internet
    |
Cloudflare Edge
    |
    ├── api-salary.neevs.io ──┐
    └── app-salary.neevs.io ──┼── Tunnel ── GitHub Runner
                              │              ├── API :8000
                              │              └── Frontend :8501
```

The frontend uses `http://localhost:8000` for API calls since both run on the same runner.

## Usage

```bash
# Start hosting (via GitHub Actions UI or CLI)
gh workflow run tunnel-hosting.yml

# With custom duration
gh workflow run tunnel-hosting.yml -f duration_hours=2

# Disable auto-restart
gh workflow run tunnel-hosting.yml -f auto_restart=false
```

## 6-Hour Timeout Handling

GitHub Actions has a 6-hour job limit. The workflow:

1. Runs for 5.5 hours by default
2. Triggers a new workflow 5 min before timeout
3. New runner takes over with fresh tunnel connection

## Limitations

- Not for production workloads
- ~1-2 min cold start when no runner active
- Requires model in S3 (won't train)
