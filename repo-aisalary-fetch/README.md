# AI Salary Data Collector

Automated tool for collecting and analyzing AI/tech job salary data from LinkedIn.

## Features

- Query LinkedIn job postings with advanced filters
- Extract salary information and job details
- Automated data collection via GitHub Actions
- Export data to JSONL format for ML training
- S3 integration for data storage

## Installation

```bash
npm install
```

## Quick Start

Run a basic job search:

```bash
npm test
```

Collect salary data:

```bash
npm run collect
```

Deduplicate collected data:

```bash
npm run deduplicate
```

## Usage

```javascript
const linkedIn = require('./index');

const queryOptions = {
  keyword: 'machine learning engineer',
  location: 'United States',
  salary: '100000',
  dateSincePosted: 'past Week',
  experienceLevel: 'mid-senior level',
  limit: '50'
};

linkedIn.query(queryOptions).then(jobs => {
  console.log(`Found ${jobs.length} jobs`);
  jobs.forEach(job => {
    console.log(`${job.position} at ${job.company} - ${job.salary}`);
  });
});
```

## Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| keyword | string | Job title or keywords to search |
| location | string | City, state, or country |
| salary | string | Minimum salary: `40000`, `60000`, `80000`, `100000`, `120000` |
| dateSincePosted | string | `past month`, `past week`, `24hr` |
| jobType | string | `full time`, `part time`, `contract`, `internship` |
| remoteFilter | string | `on site`, `remote`, `hybrid` |
| experienceLevel | string | `internship`, `entry level`, `associate`, `mid-senior level`, `director`, `executive` |
| limit | string | Number of results to return |
| sortBy | string | `recent`, `relevant` |
| page | string | Page number for pagination |

## Response Format

Each job object contains:

```javascript
{
  position: "Machine Learning Engineer",
  company: "TechCorp",
  companyLogo: "https://...",
  location: "San Francisco, CA",
  date: "2025-01-15",
  agoTime: "2 days ago",
  salary: "$150,000 - $200,000",
  jobUrl: "https://linkedin.com/jobs/..."
}
```

## Data Quality

### Deduplication

The system implements **two-stage deduplication**:

1. **During collection**: Automatically removes duplicates within each run
2. **Post-processing**: Consolidates multiple runs and removes cross-run duplicates

```bash
# Deduplicate all collected data
npm run deduplicate

# Creates: ./data/deduplicated-jobs.jsonl
```

The deduplication process:
- Uses `job_url` as unique identifier
- Keeps most recent/complete records
- Validates required fields
- Normalizes data formats
- Removes invalid entries

### Automated Collection

- **Parallel execution**: Up to 20 concurrent GitHub runners
- **85% faster**: 6 minutes vs 45 minutes for sequential collection
- Daily scheduled runs via GitHub Actions
- S3 storage for collected data
- JSONL export format for ML pipelines

See [docs/setup.md](docs/setup.md) for setup and [docs/parallel-collection.md](docs/parallel-collection.md) for parallel strategy.

## Project Structure

```
├── docs/           # Documentation
├── scripts/        # Data collection scripts
├── tests/          # Test files
└── index.js        # Core library
```

## Contributing

Fork the repository, make changes, and submit a pull request.

## License

MIT
