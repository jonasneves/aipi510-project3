const axios = require("axios");
const cheerio = require("cheerio");
const randomUseragent = require("random-useragent");

async function exploreJobFields(url) {
  try {
    const response = await axios.get(url, {
      headers: {
        "User-Agent": randomUseragent.getRandom(),
        Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      },
      timeout: 10000,
    });

    const $ = cheerio.load(response.data);

    console.log("=== AVAILABLE FIELDS ON JOB DETAIL PAGE ===\n");

    // Job description
    const description = $(".description__text, .show-more-less-html__markup").first().text().trim();
    console.log("1. Job Description:", description ? `${description.substring(0, 200)}...` : "Not found");

    // Seniority level
    const seniorityLevel = $(".description__job-criteria-item:contains('Seniority level')").find(".description__job-criteria-text").text().trim();
    console.log("\n2. Seniority Level:", seniorityLevel || "Not found");

    // Employment type
    const employmentType = $(".description__job-criteria-item:contains('Employment type')").find(".description__job-criteria-text").text().trim();
    console.log("\n3. Employment Type:", employmentType || "Not found");

    // Job function
    const jobFunction = $(".description__job-criteria-item:contains('Job function')").find(".description__job-criteria-text").text().trim();
    console.log("\n4. Job Function:", jobFunction || "Not found");

    // Industries
    const industries = $(".description__job-criteria-item:contains('Industries')").find(".description__job-criteria-text").text().trim();
    console.log("\n5. Industries:", industries || "Not found");

    // Number of applicants
    const applicants = $(".num-applicants__caption, .num-applicants__figure").text().trim();
    console.log("\n6. Number of Applicants:", applicants || "Not found");

    // Company info
    const companyInfo = $(".topcard__org-name-link, .company-name").text().trim();
    console.log("\n7. Company Info:", companyInfo || "Not found");

    // Skills (if available)
    const skills = [];
    $(".skill-item, .job-details-skill-match-status-list__skill").each((i, el) => {
      skills.push($(el).text().trim());
    });
    console.log("\n8. Skills:", skills.length > 0 ? skills.join(", ") : "Not found");

    // Benefits
    const benefits = [];
    $(".benefits-item, .job-details-benefits__benefit").each((i, el) => {
      benefits.push($(el).text().trim());
    });
    console.log("\n9. Benefits:", benefits.length > 0 ? benefits.join(", ") : "Not found");

    // Posted date
    const postedDate = $(".posted-time-ago__text, .topcard__flavor--metadata").first().text().trim();
    console.log("\n10. Posted Date:", postedDate || "Not found");

    console.log("\n=== ADDITIONAL EXTRACTABLE DATA ===");

    // All criteria items
    console.log("\nAll Job Criteria Items:");
    $(".description__job-criteria-item").each((i, el) => {
      const label = $(el).find(".description__job-criteria-subheader").text().trim();
      const value = $(el).find(".description__job-criteria-text").text().trim();
      if (label && value) {
        console.log(`  - ${label}: ${value}`);
      }
    });

  } catch (error) {
    console.log(`Error: ${error.message}`);
  }
}

const jobUrl = "https://www.linkedin.com/jobs/view/software-engineer-frontend-all-levels-at-ramp-4048167981";
exploreJobFields(jobUrl);
