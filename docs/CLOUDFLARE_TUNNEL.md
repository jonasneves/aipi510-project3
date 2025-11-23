# Cloudflare Tunnel Hosting

Host the API and frontend on GitHub Actions with Cloudflare tunnels.

## Prerequisites

1. Cloudflare account with domain
2. AWS S3 with trained model (run ML pipeline first)
3. GitHub secrets configured

## Setup

### 1. Create Cloudflare Tunnel

In Cloudflare Zero Trust dashboard:
1. Go to **Networks** > **Tunnels**
2. Click **Create a tunnel**
3. Name it (e.g., `salary-predictor`)
4. Copy the tunnel token

### 2. Configure Public Hostname

Add a public hostname for your tunnel:

| Setting | Value |
|---------|-------|
| Subdomain | `aisalary` |
| Domain | `neevs.io` (your domain) |
| Service | `HTTP://localhost:8501` |

Nginx on port 8501 handles both frontend and `/api` proxy.

### 3. Add GitHub Secrets

| Secret | Description |
|--------|-------------|
| `CLOUDFLARE_TUNNEL_TOKEN` | From step 1 |
| `AWS_ROLE_ARN` | IAM role for S3 access |

## Architecture

```
Internet
    |
Cloudflare Edge (path-based routing)
    |
aisalary.neevs.io ── Tunnel ── GitHub Runner
                                  |
                                  ├── /api/* → FastAPI :8000
                                  └── /* → npx serve :8501 (React)
```

Cloudflare routes `/api` to the API and everything else to the static frontend.

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
