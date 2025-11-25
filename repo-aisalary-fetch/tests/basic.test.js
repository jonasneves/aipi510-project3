const linkedIn = require("../index");

const queryOptions = {
  keyword: "AI",
  location: "United States",
  dateSincePosted: "",
  jobType: "",
  remoteFilter: "",
  salary: "100000",
  experienceLevel: "",
  limit: "3",
  sortBy: "recent",
  page: "0",
  has_verification: false,
  under_10_applicants: false,
  requireSalary: false,
  fetchJobDetails: true,  // Fetch detailed info including actual salary (slower)
};

linkedIn.query(queryOptions).then((response) => {
  console.log(`\n${"=".repeat(80)}`);
  console.log(`TOTAL JOBS FOUND: ${response.length}`);
  console.log(`${"=".repeat(80)}\n`);

  response.forEach((job, i) => {
    console.log(`\n[${ i + 1}] ${job.position}`);
    console.log(`${"─".repeat(80)}`);
    console.log(`Company:          ${job.company}`);
    console.log(`Location:         ${job.location}`);
    console.log(`Salary:           ${job.salary}`);
    console.log(`Seniority:        ${job.seniorityLevel || "N/A"}`);
    console.log(`Employment Type:  ${job.employmentType || "N/A"}`);
    console.log(`Job Function:     ${job.jobFunction || "N/A"}`);
    console.log(`Industries:       ${job.industries || "N/A"}`);
    console.log(`Applicants:       ${job.applicantCount || "N/A"}`);
    console.log(`Experience:       ${job.experienceYears || "N/A"}`);
    console.log(`Education:        ${job.education || "N/A"}`);
    console.log(`Skills:           ${job.skills?.length > 0 ? job.skills.join(", ") : "N/A"}`);
    console.log(`Posted:           ${job.agoTime || job.date || "N/A"}`);
    console.log(`URL:              ${job.jobUrl || "N/A"}`);
  });

  console.log(`\n${"=".repeat(80)}`);
  console.log("Sample output for ML training - all features extracted!");
  console.log(`${"=".repeat(80)}\n`);
});
