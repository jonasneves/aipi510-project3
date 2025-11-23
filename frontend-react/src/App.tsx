import { useState } from 'react'
import { DollarSign, Briefcase, MapPin, Clock, Building2, Code, Loader2, TrendingUp, TrendingDown, Github } from 'lucide-react'
import { Button } from './components/ui/button'
import { Input } from './components/ui/input'
import { Select } from './components/ui/select'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from './components/ui/card'
import { Slider } from './components/ui/slider'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const STATES = [
  'CA', 'NY', 'WA', 'TX', 'MA', 'CO', 'IL', 'GA', 'NC', 'FL',
  'PA', 'VA', 'AZ', 'OR', 'MD', 'NJ', 'OH', 'MI', 'MN', 'UT'
]

interface PredictionResult {
  predicted_salary: number
  salary_low: number
  salary_high: number
  confidence_level: string
}

function formatSalary(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(amount)
}

export default function App() {
  const [jobTitle, setJobTitle] = useState('ML Engineer')
  const [location, setLocation] = useState('CA')
  const [experience, setExperience] = useState(3)
  const [company, setCompany] = useState('')
  const [skills, setSkills] = useState('python, machine learning')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handlePredict = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_title: jobTitle,
          location: location,
          experience_years: experience,
          company: company || null,
          skills: skills ? skills.split(',').map(s => s.trim()) : null,
        }),
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get prediction')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <DollarSign className="h-6 w-6 text-primary" />
            <span className="font-semibold text-lg">AI Salary Predictor</span>
          </div>
          <a
            href="https://github.com/jonasneves/aipi510-project3"
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            <Github className="h-5 w-5" />
          </a>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold tracking-tight mb-2">
            Predict AI/ML Salaries
          </h1>
          <p className="text-muted-foreground">
            Get salary estimates based on job title, location, experience, and skills
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          {/* Input Form */}
          <Card>
            <CardHeader>
              <CardTitle className="text-xl">Job Details</CardTitle>
              <CardDescription>Enter the job information to get a salary prediction</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Job Title */}
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Briefcase className="h-4 w-4" />
                  Job Title
                </label>
                <Input
                  value={jobTitle}
                  onChange={(e) => setJobTitle(e.target.value)}
                  placeholder="e.g., Senior Data Scientist"
                />
              </div>

              {/* Location */}
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <MapPin className="h-4 w-4" />
                  Location
                </label>
                <Select value={location} onChange={(e) => setLocation(e.target.value)}>
                  {STATES.map((state) => (
                    <option key={state} value={state}>{state}</option>
                  ))}
                </Select>
              </div>

              {/* Experience */}
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Clock className="h-4 w-4" />
                  Years of Experience
                </label>
                <Slider
                  min={0}
                  max={20}
                  value={experience}
                  onChange={(e) => setExperience(Number(e.target.value))}
                  label={`${experience} years`}
                />
              </div>

              {/* Company */}
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Building2 className="h-4 w-4" />
                  Company <span className="text-muted-foreground">(optional)</span>
                </label>
                <Input
                  value={company}
                  onChange={(e) => setCompany(e.target.value)}
                  placeholder="e.g., Google"
                />
              </div>

              {/* Skills */}
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Code className="h-4 w-4" />
                  Skills <span className="text-muted-foreground">(comma-separated)</span>
                </label>
                <Input
                  value={skills}
                  onChange={(e) => setSkills(e.target.value)}
                  placeholder="e.g., pytorch, nlp, kubernetes"
                />
              </div>

              {/* Submit Button */}
              <Button
                onClick={handlePredict}
                className="w-full"
                disabled={loading || !jobTitle}
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Predicting...
                  </>
                ) : (
                  'Predict Salary'
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Results */}
          <Card>
            <CardHeader>
              <CardTitle className="text-xl">Prediction Results</CardTitle>
              <CardDescription>Salary estimate based on your inputs</CardDescription>
            </CardHeader>
            <CardContent>
              {error && (
                <div className="rounded-lg bg-red-500/10 border border-red-500/20 p-4 text-red-500">
                  {error}
                </div>
              )}

              {!result && !error && (
                <div className="text-center py-12 text-muted-foreground">
                  Enter job details and click "Predict Salary" to see results
                </div>
              )}

              {result && (
                <div className="space-y-6">
                  {/* Main Prediction */}
                  <div className="text-center py-6 rounded-lg bg-primary/5 border">
                    <p className="text-sm text-muted-foreground mb-1">Predicted Salary</p>
                    <p className="text-4xl font-bold text-primary">
                      {formatSalary(result.predicted_salary)}
                    </p>
                  </div>

                  {/* Range */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 rounded-lg bg-secondary/50 text-center">
                      <div className="flex items-center justify-center gap-1 mb-1">
                        <TrendingDown className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm text-muted-foreground">Low (90% CI)</span>
                      </div>
                      <p className="text-xl font-semibold">
                        {formatSalary(result.salary_low)}
                      </p>
                    </div>
                    <div className="p-4 rounded-lg bg-secondary/50 text-center">
                      <div className="flex items-center justify-center gap-1 mb-1">
                        <TrendingUp className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm text-muted-foreground">High (90% CI)</span>
                      </div>
                      <p className="text-xl font-semibold">
                        {formatSalary(result.salary_high)}
                      </p>
                    </div>
                  </div>

                  {/* Summary */}
                  <div className="pt-4 border-t space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Job Title</span>
                      <span className="font-medium">{jobTitle}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Location</span>
                      <span className="font-medium">{location}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Experience</span>
                      <span className="font-medium">{experience} years</span>
                    </div>
                    {company && (
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Company</span>
                        <span className="font-medium">{company}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Footer */}
        <footer className="text-center mt-12 text-sm text-muted-foreground">
          <p>Data sources: H1B filings, BLS statistics, job postings</p>
          <p className="mt-1">Built with React, Vite, and Tailwind CSS</p>
        </footer>
      </main>
    </div>
  )
}
