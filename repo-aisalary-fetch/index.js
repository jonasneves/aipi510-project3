const cheerio = require("cheerio");
const axios = require("axios");
const randomUseragent = require("random-useragent");

// ============================================================================
// CONFIGURATION CONSTANTS
// ============================================================================

const CONFIG = {
  BATCH_SIZE: 25,
  CACHE_TTL: 1000 * 60 * 60, // 1 hour
  MAX_CONSECUTIVE_ERRORS: 3,
  REQUEST_TIMEOUT: 10000,
  MIN_DELAY: 2000,
  MAX_DELAY_VARIANCE: 1000,
};

const FILTER_MAPPINGS = {
  dateSincePosted: {
    "past month": "r2592000",
    "past week": "r604800",
    "24hr": "r86400",
  },
  experienceLevel: {
    internship: "1",
    "entry level": "2",
    associate: "3",
    senior: "4",
    director: "5",
    executive: "6",
  },
  jobType: {
    "full time": "F",
    "full-time": "F",
    "part time": "P",
    "part-time": "P",
    contract: "C",
    temporary: "T",
    volunteer: "V",
    internship: "I",
  },
  remoteFilter: {
    "on-site": "1",
    "on site": "1",
    remote: "2",
    hybrid: "3",
  },
  salary: {
    40000: "1",
    60000: "2",
    80000: "3",
    100000: "4",
    120000: "5",
  },
};

const HTTP_HEADERS = {
  Accept: "application/json, text/javascript, */*; q=0.01",
  "Accept-Language": "en-US,en;q=0.9",
  "Accept-Encoding": "gzip, deflate, br",
  Referer: "https://www.linkedin.com/jobs",
  "X-Requested-With": "XMLHttpRequest",
  Connection: "keep-alive",
  "Sec-Fetch-Dest": "empty",
  "Sec-Fetch-Mode": "cors",
  "Sec-Fetch-Site": "same-origin",
  "Cache-Control": "no-cache",
  Pragma: "no-cache",
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const normalizeString = (str) => str?.trim().replace(/\s+/g, "+") || "";

const getFilterValue = (filterType, value) => {
  if (!value) return "";
  const mapping = FILTER_MAPPINGS[filterType];
  return mapping?.[value.toString().toLowerCase()] || "";
};

// ============================================================================
// CACHE IMPLEMENTATION
// ============================================================================

class JobCache {
  constructor(ttl = CONFIG.CACHE_TTL) {
    this.cache = new Map();
    this.ttl = ttl;
  }

  set(key, value) {
    this.cache.set(key, {
      data: value,
      timestamp: Date.now(),
    });
  }

  get(key) {
    const item = this.cache.get(key);
    if (!item) return null;

    if (Date.now() - item.timestamp > this.ttl) {
      this.cache.delete(key);
      return null;
    }

    return item.data;
  }

  clear() {
    const now = Date.now();
    for (const [key, value] of this.cache.entries()) {
      if (now - value.timestamp > this.ttl) {
        this.cache.delete(key);
      }
    }
  }

  get size() {
    return this.cache.size;
  }
}

const cache = new JobCache();

// ============================================================================
// QUERY CLASS
// ============================================================================

class Query {
  constructor(queryObj = {}) {
    this.host = queryObj.host || "www.linkedin.com";
    this.keyword = normalizeString(queryObj.keyword);
    this.location = normalizeString(queryObj.location);
    this.dateSincePosted = queryObj.dateSincePosted || "";
    this.jobType = queryObj.jobType || "";
    this.remoteFilter = queryObj.remoteFilter || "";
    this.salary = queryObj.salary || "";
    this.experienceLevel = queryObj.experienceLevel || "";
    this.sortBy = queryObj.sortBy || "";
    this.limit = Number(queryObj.limit) || 0;
    this.page = Number(queryObj.page) || 0;
    this.has_verification = queryObj.has_verification || false;
    this.under_10_applicants = queryObj.under_10_applicants || false;
    this.requireSalary = queryObj.requireSalary || false;
    this.fetchJobDetails = queryObj.fetchJobDetails || false;
  }

  getCacheKey() {
    return `${this.buildUrl(0)}_limit:${this.limit}`;
  }

  buildUrl(start) {
    const baseUrl = `https://${this.host}/jobs-guest/jobs/api/seeMoreJobPostings/search?`;
    const params = new URLSearchParams();

    // Add search parameters
    if (this.keyword) params.append("keywords", this.keyword);
    if (this.location) params.append("location", this.location);

    // Add filter parameters
    this.addFilterParam(params, "f_TPR", "dateSincePosted");
    this.addFilterParam(params, "f_SB2", "salary");
    this.addFilterParam(params, "f_E", "experienceLevel");
    this.addFilterParam(params, "f_WT", "remoteFilter");
    this.addFilterParam(params, "f_JT", "jobType");

    // Add boolean filters
    if (this.has_verification) params.append("f_VJ", "true");
    if (this.under_10_applicants) params.append("f_EA", "true");

    // Add pagination
    params.append("start", start + this.page * CONFIG.BATCH_SIZE);

    // Add sorting
    if (this.sortBy === "recent") params.append("sortBy", "DD");
    else if (this.sortBy === "relevant") params.append("sortBy", "R");

    return baseUrl + params.toString();
  }

  addFilterParam(params, paramName, filterType) {
    const value = getFilterValue(filterType, this[filterType]);
    if (value) params.append(paramName, value);
  }

  async getJobs() {
    const cacheKey = this.getCacheKey();
    const cachedJobs = cache.get(cacheKey);

    if (cachedJobs) {
      console.log("Returning cached results");
      return cachedJobs;
    }

    const jobs = await this.fetchAllJobs();

    if (jobs.length > 0) {
      cache.set(cacheKey, jobs);
    }

    return jobs;
  }

  async fetchAllJobs() {
    let allJobs = [];
    let start = 0;
    let hasMore = true;
    let consecutiveErrors = 0;

    console.log(this.buildUrl(0));
    console.log(this.getCacheKey());

    while (hasMore) {
      try {
        const jobs = await this.fetchJobBatch(start);

        if (!jobs || jobs.length === 0) {
          hasMore = false;
          break;
        }

        allJobs.push(...jobs);
        console.log(`Fetched ${jobs.length} jobs. Total: ${allJobs.length}`);

        if (this.limit && allJobs.length >= this.limit) {
          allJobs = allJobs.slice(0, this.limit);
          break;
        }

        consecutiveErrors = 0;
        start += CONFIG.BATCH_SIZE;

        await delay(CONFIG.MIN_DELAY + Math.random() * CONFIG.MAX_DELAY_VARIANCE);
      } catch (error) {
        consecutiveErrors++;
        console.error(
          `Error fetching batch (attempt ${consecutiveErrors}):`,
          error.message
        );

        if (consecutiveErrors >= CONFIG.MAX_CONSECUTIVE_ERRORS) {
          console.log("Max consecutive errors reached. Stopping.");
          break;
        }

        await delay(Math.pow(2, consecutiveErrors) * 1000);
      }
    }

    return allJobs;
  }

  async fetchJobBatch(start) {
    const headers = {
      "User-Agent": randomUseragent.getRandom(),
      ...HTTP_HEADERS,
    };

    try {
      const response = await axios.get(this.buildUrl(start), {
        headers,
        validateStatus: (status) => status === 200,
        timeout: CONFIG.REQUEST_TIMEOUT,
      });

      let jobs = parseJobList(response.data, this.requireSalary);

      // Fetch detailed info if requested
      if (this.fetchJobDetails && jobs.length > 0) {
        jobs = await this.enrichJobsWithDetails(jobs);
      }

      return jobs;
    } catch (error) {
      if (error.response?.status === 429) {
        throw new Error("Rate limit reached");
      }
      throw error;
    }
  }

  async enrichJobsWithDetails(jobs) {
    const enrichedJobs = [];

    for (const job of jobs) {
      try {
        const details = await this.fetchJobDetail(job.jobUrl);
        enrichedJobs.push({ ...job, ...details });

        // Small delay between detail requests to avoid rate limiting
        await delay(1000 + Math.random() * 500);
      } catch (error) {
        console.warn(`Failed to fetch details for ${job.position}:`, error.message);
        enrichedJobs.push(job); // Keep original job if detail fetch fails
      }
    }

    return enrichedJobs;
  }

  async fetchJobDetail(jobUrl) {
    if (!jobUrl) return {};

    const headers = {
      "User-Agent": randomUseragent.getRandom(),
      Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      "Accept-Language": "en-US,en;q=0.9",
    };

    try {
      const response = await axios.get(jobUrl, {
        headers,
        timeout: CONFIG.REQUEST_TIMEOUT,
      });

      const $ = cheerio.load(response.data);

      // Extract salary
      const salaryElement = $(".compensation__salary, .salary").first();
      const detailedSalary = salaryElement.text().trim();

      // Extract job criteria fields
      const seniorityLevel = this.extractCriteria($, "Seniority level");
      const employmentType = this.extractCriteria($, "Employment type");
      const jobFunction = this.extractCriteria($, "Job function");
      const industries = this.extractCriteria($, "Industries");

      // Extract number of applicants
      const applicantsText = $(".num-applicants__caption, .num-applicants__figure").text().trim();
      const applicantCount = this.parseApplicantCount(applicantsText);

      // Extract job description
      const description = $(".description__text, .show-more-less-html__markup").first().text().trim();

      // Parse skills from description
      const skills = this.extractSkills(description);

      // Parse experience requirements
      const experienceYears = this.extractExperience(description);

      // Parse education requirements
      const education = this.extractEducation(description);

      return {
        salary: detailedSalary || "Not specified",
        seniorityLevel: seniorityLevel || "Not specified",
        employmentType: employmentType || "Not specified",
        jobFunction: jobFunction || "Not specified",
        industries: industries || "Not specified",
        applicantCount: applicantCount,
        description: description || "",
        skills: skills,
        experienceYears: experienceYears,
        education: education,
      };
    } catch (error) {
      return {};
    }
  }

  extractCriteria($, criteriaName) {
    return $(`.description__job-criteria-item:contains('${criteriaName}')`)
      .find(".description__job-criteria-text")
      .text()
      .trim();
  }

  parseApplicantCount(text) {
    if (!text) return 0;

    // Handle "Over X applicants" or "X applicants"
    const match = text.match(/over\s+(\d+)|(\d+)\s+applicant/i);
    if (match) {
      return parseInt(match[1] || match[2]);
    }

    // Handle ranges like "100-200 applicants"
    const rangeMatch = text.match(/(\d+)\s*-\s*(\d+)/);
    if (rangeMatch) {
      return parseInt(rangeMatch[2]); // Return upper bound
    }

    return 0;
  }

  extractSkills(description) {
    if (!description) return [];

    const skillKeywords = [
      // AI/ML
      "machine learning", "deep learning", "neural networks", "nlp", "computer vision",
      "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",

      // Programming
      "python", "java", "javascript", "c\\+\\+", "scala", "r\\b", "go\\b", "rust",

      // Cloud & Infrastructure
      "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "kafka",

      // Databases
      "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",

      // Other
      "spark", "hadoop", "airflow", "mlflow", "git", "ci/cd", "rest api", "graphql"
    ];

    const foundSkills = [];
    const lowerDesc = description.toLowerCase();

    skillKeywords.forEach(skill => {
      const regex = new RegExp(`\\b${skill}\\b`, "i");
      if (regex.test(lowerDesc)) {
        foundSkills.push(skill.replace(/\\b/g, "").replace(/\\\+\\\+/, "++"));
      }
    });

    return [...new Set(foundSkills)]; // Remove duplicates
  }

  extractExperience(description) {
    if (!description) return "Not specified";

    // Match patterns like "3+ years", "5-7 years", "3 to 5 years"
    const patterns = [
      /(\d+)\+?\s*(?:to|\-)\s*(\d+)\s*(?:years?|yrs?)/i,
      /(\d+)\+\s*(?:years?|yrs?)/i,
      /minimum\s+(?:of\s+)?(\d+)\s*(?:years?|yrs?)/i,
      /at least\s+(\d+)\s*(?:years?|yrs?)/i,
    ];

    for (const pattern of patterns) {
      const match = description.match(pattern);
      if (match) {
        if (match[2]) {
          return `${match[1]}-${match[2]} years`;
        }
        return `${match[1]}+ years`;
      }
    }

    return "Not specified";
  }

  extractEducation(description) {
    if (!description) return "Not specified";

    const lowerDesc = description.toLowerCase();

    if (lowerDesc.includes("ph.d") || lowerDesc.includes("phd") || lowerDesc.includes("doctorate")) {
      return "PhD";
    }
    if (lowerDesc.includes("master") || lowerDesc.includes("m.s.") || lowerDesc.includes("graduate degree")) {
      return "Masters";
    }
    if (lowerDesc.includes("bachelor") || lowerDesc.includes("b.s.") || lowerDesc.includes("b.a.") || lowerDesc.includes("undergraduate")) {
      return "Bachelors";
    }

    return "Not specified";
  }
}

// ============================================================================
// PARSING FUNCTIONS
// ============================================================================

function parseJobList(jobData, requireSalary = false) {
  try {
    const $ = cheerio.load(jobData);
    const jobs = $("li");

    return jobs
      .map((index, element) => parseJobElement($, element, index, requireSalary))
      .get()
      .filter(Boolean);
  } catch (error) {
    console.error("Error parsing job list:", error);
    return [];
  }
}

function parseJobElement($, element, index, requireSalary = false) {
  try {
    const job = $(element);

    const position = job.find(".base-search-card__title").text().trim();
    const company = job.find(".base-search-card__subtitle").text().trim();

    if (!position || !company) {
      return null;
    }

    const location = job.find(".job-search-card__location").text().trim();
    const dateElement = job.find("time");
    const date = dateElement.attr("datetime");
    const salaryText = job
      .find(".job-search-card__salary-info")
      .text()
      .trim()
      .replace(/\s+/g, " ");

    const salary = salaryText || "Not specified";

    // Filter out jobs without salary if requireSalary is true
    if (requireSalary && salary === "Not specified") {
      return null;
    }

    const jobUrl = job.find(".base-card__full-link").attr("href");
    const companyLogo = job
      .find(".artdeco-entity-image")
      .attr("data-delayed-url");
    const agoTime = job.find(".job-search-card__listdate").text().trim();

    return {
      position,
      company,
      location,
      date,
      salary,
      jobUrl: jobUrl || "",
      companyLogo: companyLogo || "",
      agoTime: agoTime || "",
    };
  } catch (err) {
    console.warn(`Error parsing job at index ${index}:`, err.message);
    return null;
  }
}

// ============================================================================
// PUBLIC API
// ============================================================================

module.exports.query = (queryObject) => {
  const query = new Query(queryObject);
  return query.getJobs();
};

module.exports.JobCache = JobCache;
module.exports.clearCache = () => cache.clear();
module.exports.getCacheSize = () => cache.size;
