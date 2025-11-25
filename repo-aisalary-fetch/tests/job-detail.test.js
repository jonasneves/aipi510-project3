const axios = require("axios");
const cheerio = require("cheerio");
const randomUseragent = require("random-useragent");

async function fetchJobDetail(url) {
  try {
    const response = await axios.get(url, {
      headers: {
        "User-Agent": randomUseragent.getRandom(),
        Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      },
      timeout: 10000,
    });

    const $ = cheerio.load(response.data);

    // Try to find salary information - common selectors
    const selectors = [
      ".compensation__salary",
      ".salary-main-rail__salary-info",
      ".salary",
      "[class*='salary']",
      "[class*='compensation']",
    ];

    console.log("Looking for salary information...\n");

    selectors.forEach(selector => {
      const elements = $(selector);
      if (elements.length > 0) {
        console.log(`Found with selector "${selector}":`);
        elements.each((i, el) => {
          console.log(`  ${$(el).text().trim()}`);
        });
      }
    });

    // Also search for text containing dollar signs
    const bodyText = $("body").text();
    const salaryMatches = bodyText.match(/\$[\d,]+(?:\s*-\s*\$[\d,]+)?(?:\s*(?:per|\/)\s*(?:year|yr|hour|hr|month|mo))?/gi);

    if (salaryMatches) {
      console.log("\nFound potential salary patterns in text:");
      salaryMatches.slice(0, 10).forEach(match => {
        console.log(`  ${match}`);
      });
    }

  } catch (error) {
    console.log(`Error: ${error.message}`);
  }
}

const jobUrl = "https://www.linkedin.com/jobs/view/ai-ml-software-engineer-at-cooperidge-consulting-firm-4338894279";
fetchJobDetail(jobUrl);
