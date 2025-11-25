const axios = require("axios");
const cheerio = require("cheerio");
const randomUseragent = require("random-useragent");

async function testEndpoint(url, name) {
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

    console.log(`\n${name}:`);
    console.log(`- URL: ${url}`);
    console.log(`- Response length: ${response.data.length} chars`);
    console.log(`- Jobs found: ${jobs.length}`);

    // Check for salary info
    let jobsWithSalary = 0;
    jobs.each((i, el) => {
      const salary = $(el).find(".job-search-card__salary-info").text().trim();
      if (salary) jobsWithSalary++;
    });
    console.log(`- Jobs with salary: ${jobsWithSalary}`);

  } catch (error) {
    console.log(`\n${name}: ERROR - ${error.message}`);
  }
}

(async () => {
  const keyword = "ai nvidia";

  // Old endpoint (current)
  const oldUrl = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords=ai+nvidia&location=United+States&start=0";

  // New endpoint (better?)
  const newUrl = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search-results/?keywords=ai%20nvidia&origin=JOB_COLLECTION_PAGE_SEARCH_BUTTON&start=0";

  await testEndpoint(oldUrl, "Current Endpoint (search)");
  await testEndpoint(newUrl, "New Endpoint (search-results)");
})();
