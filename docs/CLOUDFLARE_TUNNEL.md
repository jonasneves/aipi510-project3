# Cloudflare Tunnel Setup for GitHub Actions Hosting

Host the API and frontend on GitHub Actions with Cloudflare tunnels for public access.

## Prerequisites

1. Cloudflare account (free tier works)
2. Domain added to Cloudflare (or use `*.trycloudflare.com` for testing)

## Setup Steps

### 1. Create Cloudflare Tunnel

```bash
# Install cloudflared locally
brew install cloudflare/cloudflare/cloudflared  # macOS
# or download from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

# Login to Cloudflare
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create salary-predictor

# Note the tunnel ID and credentials file path
```

### 2. Configure DNS (if using custom domain)

```bash
# Create DNS records pointing to tunnel
cloudflared tunnel route dns salary-predictor api.yourdomain.com
cloudflared tunnel route dns salary-predictor app.yourdomain.com
```

### 3. Add GitHub Secrets

Add these secrets to your repository (Settings > Secrets > Actions):

| Secret | Description |
|--------|-------------|
| `CLOUDFLARE_TUNNEL_TOKEN` | Tunnel token from `cloudflared tunnel token <tunnel-id>` |
| `CLOUDFLARE_TUNNEL_ID` | Your tunnel ID |

### 4. Quick Start (No Custom Domain)

For testing without a domain, the workflow uses `trycloudflare.com` quick tunnels.
No secrets needed - just run the workflow.

## Architecture

```
Internet
    |
    v
Cloudflare Edge (DDoS protection, caching)
    |
    v
Cloudflared Tunnel (outbound-only, no open ports)
    |
    v
GitHub Actions Runner
    |-- API (port 8000)
    |-- Frontend (port 8501)
```

## Handling 6-Hour Timeout

GitHub Actions has a 6-hour job limit. The workflow handles this by:

1. Running services for ~5.5 hours
2. Triggering a new workflow run before timeout
3. Waiting for new runner to establish tunnel
4. Gracefully shutting down old runner

This provides near-zero-downtime continuous hosting.

## Cost

- GitHub Actions: 2000 free minutes/month (public repos unlimited)
- Cloudflare Tunnel: Free tier included
- Total: $0 for hobby/demo projects

## Limitations

- Not for production workloads
- Cold start when no runner is active (~1-2 min)
- Dependent on GitHub Actions availability
