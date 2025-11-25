const fs = require("fs");
const path = require("path");
const linkedIn = require("../index");

// Configuration
const OUTPUT_DIR = "./data";
const TIMESTAMP = new Date().toISOString().replace(/[:.]/g, "-");
const OUTPUT_FILE = path.join(OUTPUT_DIR, `ai-jobs-${TIMESTAMP}.jsonl`);

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Create write stream for incremental saving
const writeStream = fs.createWriteStream(OUTPUT_FILE, { flags: "a" });

console.log(`\n${"=".repeat(80)}`);
console.log(`AI SALARY DATA COLLECTION`);
console.log(`Output: ${OUTPUT_FILE}`);
console.log(`${"=".repeat(80)}\n`);

// Search configuration
const searchQueries = [
  {
    keyword: "machine learning engineer",
    location: "United States",
    salary: "100000",
    limit: "50",
  },
  {
    keyword: "data scientist",
    location: "United States",
    salary: "100000",
    limit: "50",
  },
  {
    keyword: "AI engineer",
    location: "United States",
    salary: "100000",
    limit: "50",
  },
];

const baseQueryOptions = {
  dateSincePosted: "",
  jobType: "",
  remoteFilter: "",
  experienceLevel: "",
  sortBy: "recent",
  page: "0",
  has_verification: false,
  under_10_applicants: false,
  requireSalary: false,
  fetchJobDetails: true, // Get all ML features
};

let totalJobsCollected = 0;
let jobsWithSalary = 0;
const seenJobUrls = new Set(); // Track job URLs within this run

// Function to save job to file
function saveJob(job) {
  // Skip duplicates within this collection run
  if (seenJobUrls.has(job.jobUrl)) {
    return false;
  }
  seenJobUrls.add(job.jobUrl);
  const jobData = {
    // Metadata
    collected_at: new Date().toISOString(),

    // Basic info
    position: job.position,
    company: job.company,
    location: job.location,
    posted_date: job.date,
    job_url: job.jobUrl,

    // Target variable
    salary: job.salary,

    // Features for ML model
    seniority_level: job.seniorityLevel,
    employment_type: job.employmentType,
    job_function: job.jobFunction,
    industries: job.industries,
    applicant_count: job.applicantCount,
    experience_years: job.experienceYears,
    education: job.education,
    skills: job.skills || [],

    // Additional context
    company_logo: job.companyLogo,
    description: job.description ? job.description.substring(0, 500) : "", // Truncate for size
  };

  // Write as JSON line
  writeStream.write(JSON.stringify(jobData) + "\n");

  totalJobsCollected++;
  if (job.salary && job.salary !== "Not specified") {
    jobsWithSalary++;
  }

  // Progress indicator
  if (totalJobsCollected % 10 === 0) {
    console.log(`Progress: ${totalJobsCollected} jobs collected (${jobsWithSalary} with salary)`);
  }

  return true;
}

// Main collection function
async function collectData() {
  console.log(`Starting data collection for ${searchQueries.length} search queries...\n`);

  for (const [index, searchQuery] of searchQueries.entries()) {
    console.log(`\n[${ index + 1}/${searchQueries.length}] Searching: "${searchQuery.keyword}" in ${searchQuery.location}`);

    const queryOptions = {
      ...baseQueryOptions,
      ...searchQuery,
    };

    try {
      const jobs = await linkedIn.query(queryOptions);

      console.log(`  Found ${jobs.length} jobs`);

      // Save each job incrementally (skipping duplicates)
      let saved = 0;
      let skipped = 0;
      jobs.forEach(job => {
        if (saveJob(job)) {
          saved++;
        } else {
          skipped++;
        }
      });

      if (skipped > 0) {
        console.log(`  Saved ${saved} unique jobs (skipped ${skipped} duplicates)`);
      }

      // Delay between queries to avoid rate limiting
      if (index < searchQueries.length - 1) {
        console.log(`  Waiting before next query...`);
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    } catch (error) {
      console.error(`  Error: ${error.message}`);
    }
  }

  // Close write stream
  writeStream.end();

  console.log(`\n${"=".repeat(80)}`);
  console.log(`DATA COLLECTION COMPLETE`);
  console.log(`${"=".repeat(80)}`);
  console.log(`Total jobs collected: ${totalJobsCollected}`);
  console.log(`Jobs with salary: ${jobsWithSalary} (${((jobsWithSalary/totalJobsCollected)*100).toFixed(1)}%)`);
  console.log(`Output file: ${OUTPUT_FILE}`);
  console.log(`File size: ${(fs.statSync(OUTPUT_FILE).size / 1024).toFixed(2)} KB`);
  console.log(`${"=".repeat(80)}\n`);
}

// Run collection
collectData().catch(error => {
  console.error("Fatal error:", error);
  process.exit(1);
});
