const axios = require("axios");
const cheerio = require("cheerio");
const randomUseragent = require("random-useragent");

async function getJobs(url, name) {
  try {
    const response = await axios.get(url, {
      headers: {
        "User-Agent": randomUseragent.getRandom(),
        Accept: "application/json, text/javascript, */*; q=0.01",
      },
      timeout: 10000,
    });

    const $ = cheerio.load(response.data);
    const jobs = $("li");

    const jobList = [];
    jobs.each((i, el) => {
      const job = $(el);
      const position = job.find(".base-search-card__title").text().trim();
      const company = job.find(".base-search-card__subtitle").text().trim();
      const salary = job.find(".job-search-card__salary-info").text().trim();

      if (position && company) {
        jobList.push({ position, company, salary: salary || "Not specified" });
      }
    });

    console.log(`\n${name} (${jobList.length} jobs):`);
    console.log(JSON.stringify(jobList, null, 2));

    return jobList;

  } catch (error) {
    console.log(`\n${name}: ERROR - ${error.message}`);
    return [];
  }
}

(async () => {
  // Old endpoint (current)
  const oldUrl = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords=ai+nvidia&location=United+States&start=0";

  // New endpoint (better?)
  const newUrl = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search-results/?keywords=ai%20nvidia&origin=JOB_COLLECTION_PAGE_SEARCH_BUTTON&start=0";

  const oldJobs = await getJobs(oldUrl, "Current Endpoint");
  await new Promise(r => setTimeout(r, 2000)); // Wait between requests
  const newJobs = await getJobs(newUrl, "New Endpoint");

  console.log("\n=== COMPARISON ===");
  console.log(`Old endpoint: ${oldJobs.length} jobs`);
  console.log(`New endpoint: ${newJobs.length} jobs`);

  // Check if jobs are the same
  const oldTitles = oldJobs.map(j => j.position);
  const newTitles = newJobs.map(j => j.position);

  const uniqueToOld = oldTitles.filter(t => !newTitles.includes(t));
  const uniqueToNew = newTitles.filter(t => !oldTitles.includes(t));

  if (uniqueToOld.length > 0) {
    console.log("\nOnly in OLD endpoint:", uniqueToOld);
  }
  if (uniqueToNew.length > 0) {
    console.log("\nOnly in NEW endpoint:", uniqueToNew);
  }
  if (uniqueToOld.length === 0 && uniqueToNew.length === 0) {
    console.log("\nBoth endpoints return the SAME jobs!");
  }
})();
