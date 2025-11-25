const fs = require("fs");
const path = require("path");

/**
 * Deduplication and data cleaning script
 *
 * Usage:
 *   node scripts/deduplicate-data.js [input-dir] [output-file]
 *
 * Example:
 *   node scripts/deduplicate-data.js ./data ./data/cleaned-jobs.jsonl
 */

const INPUT_DIR = process.argv[2] || "./data";
const OUTPUT_FILE = process.argv[3] || "./data/deduplicated-jobs.jsonl";

console.log(`\n${"=".repeat(80)}`);
console.log(`DATA DEDUPLICATION AND CLEANING`);
console.log(`${"=".repeat(80)}\n`);
console.log(`Input directory: ${INPUT_DIR}`);
console.log(`Output file: ${OUTPUT_FILE}\n`);

// Statistics
const stats = {
  filesProcessed: 0,
  totalRecords: 0,
  duplicatesRemoved: 0,
  invalidRecords: 0,
  cleanedRecords: 0,
};

// Store unique jobs by job_url
const uniqueJobs = new Map();

/**
 * Validate and clean a job record
 */
function cleanJobRecord(job) {
  // Required fields validation
  if (!job.job_url || !job.position || !job.company) {
    return null;
  }

  // Clean and normalize data
  const cleaned = {
    ...job,

    // Normalize salary field
    salary: job.salary === "" || job.salary === "Not specified"
      ? null
      : job.salary,

    // Ensure arrays are arrays
    skills: Array.isArray(job.skills) ? job.skills : [],

    // Normalize dates
    collected_at: job.collected_at || new Date().toISOString(),

    // Remove excessive whitespace from text fields
    position: job.position?.trim(),
    company: job.company?.trim(),
    location: job.location?.trim(),
    description: job.description?.trim(),
  };

  return cleaned;
}

/**
 * Determine which duplicate to keep (most recent collection)
 */
function chooseBestRecord(existing, newRecord) {
  // Keep the one with more complete data
  const existingSalary = existing.salary && existing.salary !== null;
  const newSalary = newRecord.salary && newRecord.salary !== null;

  // Prefer record with salary data
  if (newSalary && !existingSalary) return newRecord;
  if (existingSalary && !newSalary) return existing;

  // Keep more recently collected
  if (new Date(newRecord.collected_at) > new Date(existing.collected_at)) {
    return newRecord;
  }

  return existing;
}

/**
 * Process a single JSONL file
 */
function processFile(filePath) {
  console.log(`Processing: ${path.basename(filePath)}`);

  const content = fs.readFileSync(filePath, "utf8");
  const lines = content.trim().split("\n").filter(line => line.length > 0);

  let fileRecords = 0;
  let fileDuplicates = 0;
  let fileInvalid = 0;

  lines.forEach(line => {
    stats.totalRecords++;
    fileRecords++;

    try {
      const job = JSON.parse(line);
      const cleaned = cleanJobRecord(job);

      if (!cleaned) {
        stats.invalidRecords++;
        fileInvalid++;
        return;
      }

      const jobUrl = cleaned.job_url;

      if (uniqueJobs.has(jobUrl)) {
        // Duplicate found - choose best record
        const existing = uniqueJobs.get(jobUrl);
        const best = chooseBestRecord(existing, cleaned);
        uniqueJobs.set(jobUrl, best);
        stats.duplicatesRemoved++;
        fileDuplicates++;
      } else {
        uniqueJobs.set(jobUrl, cleaned);
      }

    } catch (error) {
      console.error(`  Invalid JSON: ${error.message}`);
      stats.invalidRecords++;
      fileInvalid++;
    }
  });

  console.log(`  Records: ${fileRecords}, Duplicates: ${fileDuplicates}, Invalid: ${fileInvalid}`);
  stats.filesProcessed++;
}

/**
 * Main execution
 */
function main() {
  // Find all JSONL files in input directory
  const files = fs.readdirSync(INPUT_DIR)
    .filter(file => file.endsWith(".jsonl") && file.startsWith("ai-jobs-"))
    .map(file => path.join(INPUT_DIR, file))
    .sort();

  if (files.length === 0) {
    console.error("No JSONL files found in input directory");
    process.exit(1);
  }

  console.log(`Found ${files.length} files to process\n`);

  // Process each file
  files.forEach(processFile);

  // Write deduplicated data
  console.log(`\nWriting deduplicated data...`);

  const outputDir = path.dirname(OUTPUT_FILE);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const writeStream = fs.createWriteStream(OUTPUT_FILE);

  uniqueJobs.forEach(job => {
    writeStream.write(JSON.stringify(job) + "\n");
    stats.cleanedRecords++;
  });

  writeStream.end();

  // Print summary
  console.log(`\n${"=".repeat(80)}`);
  console.log(`DEDUPLICATION COMPLETE`);
  console.log(`${"=".repeat(80)}`);
  console.log(`Files processed: ${stats.filesProcessed}`);
  console.log(`Total records: ${stats.totalRecords}`);
  console.log(`Duplicates removed: ${stats.duplicatesRemoved}`);
  console.log(`Invalid records: ${stats.invalidRecords}`);
  console.log(`Clean unique records: ${stats.cleanedRecords}`);
  console.log(`Deduplication rate: ${((stats.duplicatesRemoved / stats.totalRecords) * 100).toFixed(1)}%`);
  console.log(`Output file: ${OUTPUT_FILE}`);
  console.log(`File size: ${(fs.statSync(OUTPUT_FILE).size / 1024).toFixed(2)} KB`);
  console.log(`${"=".repeat(80)}\n`);
}

main();
