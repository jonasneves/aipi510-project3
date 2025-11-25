const linkedIn = require("../index");

// Create a query instance to test the fetchJobDetail method
const Query = linkedIn.Query || class Query {
  async fetchJobDetail(jobUrl) {
    const axios = require("axios");
    const cheerio = require("cheerio");
    const randomUseragent = require("random-useragent");

    const headers = {
      "User-Agent": randomUseragent.getRandom(),
      Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
      "Accept-Language": "en-US,en;q=0.9",
    };

    try {
      const response = await axios.get(jobUrl, {
        headers,
        timeout: 10000,
      });

      const $ = cheerio.load(response.data);
      const salaryElement = $(".compensation__salary, .salary").first();
      const detailedSalary = salaryElement.text().trim();

      console.log("Salary found:", detailedSalary || "Not specified");
      return { salary: detailedSalary || "Not specified" };
    } catch (error) {
      console.log("Error:", error.message);
      return {};
    }
  }
};

const query = new Query({});
const testUrl = "https://www.linkedin.com/jobs/view/ai-ml-software-engineer-at-cooperidge-consulting-firm-4338894279";

console.log("Testing job detail fetch for:", testUrl);
query.fetchJobDetail(testUrl);
