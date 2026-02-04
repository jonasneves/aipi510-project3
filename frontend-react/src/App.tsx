import { useState, useEffect, useCallback } from 'react'
import { Briefcase, MapPin, Clock, Building2, Code, Loader2, TrendingUp, MapPinned, Sparkles, FileText, X } from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, ReferenceLine } from 'recharts'
import { Input } from './components/ui/input'
import { Select } from './components/ui/select'
import { Card, CardHeader, CardTitle, CardContent } from './components/ui/card'
import { Slider } from './components/ui/slider'
import { Tabs, TabsList, TabsTrigger, TabsContent } from './components/ui/tabs'

// API URL configuration:
// - localhost: local dev server
// - github.io: Cloudflare tunnel at aisalary.neevs.io
// - aisalary.neevs.io: Cloudflare routes /api to backend
const getApiUrl = () => {
  if (import.meta.env.VITE_API_URL) return import.meta.env.VITE_API_URL
  if (window.location.hostname === 'localhost') return 'http://localhost:8000/api'
  if (window.location.hostname.includes('github.io')) return 'https://aisalary.neevs.io/api'
  return '/api' // Default for aisalary.neevs.io
}

const API_URL = getApiUrl()

// Static fallback data for GitHub Pages deployment
const FALLBACK_OPTIONS: Options = {
  job_titles: [
    'ML Engineer', 'Data Scientist', 'AI Researcher', 'ML Researcher',
    'Deep Learning Engineer', 'Computer Vision Engineer', 'NLP Engineer',
    'Data Analyst', 'AI Engineer', 'Research Scientist'
  ],
  locations: [
    { code: 'CA', name: 'California' },
    { code: 'NY', name: 'New York' },
    { code: 'WA', name: 'Washington' },
    { code: 'TX', name: 'Texas' },
    { code: 'MA', name: 'Massachusetts' },
    { code: 'NC', name: 'North Carolina' },
    { code: 'IL', name: 'Illinois' },
    { code: 'GA', name: 'Georgia' },
    { code: 'VA', name: 'Virginia' },
    { code: 'CO', name: 'Colorado' }
  ],
  skills: [
    'Python', 'Machine Learning', 'Deep Learning', 'PyTorch', 'TensorFlow',
    'Computer Vision', 'NLP', 'AWS', 'Docker', 'Kubernetes', 'SQL', 'Spark'
  ]
}


interface Options {
  job_titles: string[]
  locations: { code: string; name: string }[]
  skills: string[]
}

interface FeatureFactor {
  name: string
  impact: string
  description: string
}

interface PredictionResult {
  predicted_salary: number
  salary_low: number
  salary_high: number
  confidence_level: string
  top_factors: FeatureFactor[]
}

interface ResumeParseResult {
  job_title: string | null
  location: string | null
  experience_years: number
  skills: string[]
  success: boolean
  message: string
}

function formatSalary(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(amount)
}

// Debounce hook
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value)

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])

  return debouncedValue
}

// Generate bell curve data points
function generateBellCurve(low: number, median: number, high: number) {
  const points = []
  const stdDev = (high - low) / 3.29 // 90% CI = ~3.29 std deviations
  const numPoints = 100

  for (let i = 0; i < numPoints; i++) {
    const x = low - stdDev + (i / (numPoints - 1)) * (high - low + 2 * stdDev)
    const z = (x - median) / stdDev
    const y = Math.exp(-0.5 * z * z) / (stdDev * Math.sqrt(2 * Math.PI))
    points.push({
      salary: x,
      density: y,
      inRange: x >= low && x <= high
    })
  }

  return points
}

export default function App() {
  const [options, setOptions] = useState<Options | null>(null)
  const [jobTitle, setJobTitle] = useState('ML Engineer')
  const [location, setLocation] = useState('NC')
  const [experience, setExperience] = useState(3)
  const [company, setCompany] = useState('')
  const [selectedSkills, setSelectedSkills] = useState<string[]>(['Python', 'Machine Learning'])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Resume upload state
  const [inputMode, setInputMode] = useState<'upload' | 'manual'>('upload')
  const [isDragging, setIsDragging] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [parsing, setParsing] = useState(false)
  const [parseMessage, setParseMessage] = useState<string | null>(null)

  // Create debounced values for auto-predict
  const debouncedJobTitle = useDebounce(jobTitle, 300)
  const debouncedLocation = useDebounce(location, 300)
  const debouncedExperience = useDebounce(experience, 300)
  const debouncedCompany = useDebounce(company, 500)
  const debouncedSkills = useDebounce(selectedSkills, 300)

  // Fetch options on mount
  useEffect(() => {
    fetch(`${API_URL}/options`)
      .then(res => {
        if (!res.ok) throw new Error('API not available')
        return res.json()
      })
      .then(data => setOptions(data))
      .catch(err => {
        console.error('Failed to fetch options:', err)
        setOptions(FALLBACK_OPTIONS)
        setError('API server offline. Predictions unavailable until the server is restarted.')
      })
  }, [])

  // Auto-predict when inputs change (in manual mode or after resume parsed)
  const predict = useCallback(async () => {
    if (!jobTitle || !location) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_title: jobTitle,
          location: location,
          experience_years: experience,
          company: company || null,
          skills: selectedSkills.length > 0 ? selectedSkills : null,
        }),
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError('API server offline. Please try again when the server is running.')
    } finally {
      setLoading(false)
    }
  }, [jobTitle, location, experience, company, selectedSkills])

  // Auto-predict on debounced changes (only in manual mode or after upload)
  useEffect(() => {
    if (inputMode === 'manual' || uploadedFile) {
      predict()
    }
  }, [debouncedJobTitle, debouncedLocation, debouncedExperience, debouncedCompany, debouncedSkills, inputMode, uploadedFile, predict])

  const toggleSkill = (skill: string) => {
    setSelectedSkills(prev =>
      prev.includes(skill)
        ? prev.filter(s => s !== skill)
        : [...prev, skill]
    )
  }

  // File upload handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) {
      await processFile(file)
    }
  }

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      await processFile(file)
    }
  }

  const processFile = async (file: File) => {
    const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
    if (!validTypes.includes(file.type) && !file.name.endsWith('.pdf') && !file.name.endsWith('.docx')) {
      setError('Please upload a PDF or DOCX file')
      return
    }

    setUploadedFile(file)
    setParsing(true)
    setError(null)
    setParseMessage(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_URL}/parse-resume`, {
        method: 'POST',
        body: formData,
      })

      const data: ResumeParseResult = await response.json()

      if (data.success) {
        // Update form with parsed data
        if (data.job_title) setJobTitle(data.job_title)
        if (data.location) setLocation(data.location)
        if (data.experience_years) setExperience(data.experience_years)
        if (data.skills.length > 0) setSelectedSkills(data.skills)
        setParseMessage(data.message)
      } else {
        setError(data.message)
      }
    } catch (err) {
      setError('API server offline. Resume parsing unavailable.')
    } finally {
      setParsing(false)
    }
  }

  const clearFile = () => {
    setUploadedFile(null)
    setParseMessage(null)
    setResult(null)
  }

  // Generate bell curve data
  const bellCurveData = result
    ? generateBellCurve(result.salary_low, result.predicted_salary, result.salary_high)
    : null

  return (
    <div className="min-h-screen relative">
      {/* Background Image */}
      <div
        className="fixed inset-0 z-0 opacity-30"
        style={{
          backgroundImage: 'url(/background.png)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
        }}
      />

      {/* Content */}
      <div className="relative z-10">
        {/* Header */}
        <header className="border-b border-border/50 backdrop-blur-sm">
          <div className="container mx-auto px-4 py-4 flex items-center gap-3">
            <img src="/logo.png" alt="AI Salary Predictor" className="h-8 w-8" />
            <div>
              <span className="font-semibold text-lg">AI Salary Predictor</span>
              <p className="text-xs text-muted-foreground">Data-driven insights from H1B, LinkedIn Jobs, and Adzuna</p>
            </div>
          </div>
        </header>


        {/* Hero Section */}
        <div className="text-center py-12 px-4">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4">
            Stop Guessing Your <span className="text-gradient">AI Salary.</span>
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Get instant, data-backed salary estimates with confidence intervals.
          </p>
        </div>

        {/* Main Content */}
        <main className="container mx-auto px-4 pb-12 max-w-5xl">
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Input Card */}
            <Card className="bg-gradient-card border-border/50">
              <CardContent className="p-0">
                <Tabs value={inputMode} onValueChange={(v) => setInputMode(v as 'upload' | 'manual')}>
                  <TabsList className="w-full rounded-t-lg rounded-b-none">
                    <TabsTrigger value="upload">Upload Resume</TabsTrigger>
                    <TabsTrigger value="manual">Manual Entry</TabsTrigger>
                  </TabsList>

                  <div className="p-6">
                    <TabsContent value="upload" className="mt-0">
                      {!uploadedFile && (
                        <div
                          onDragOver={handleDragOver}
                          onDragLeave={handleDragLeave}
                          onDrop={handleDrop}
                          className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-all ${
                            isDragging
                              ? 'border-primary bg-primary/10'
                              : 'border-muted-foreground/25 hover:border-primary/50 bg-muted/30'
                          }`}
                        >
                          <input
                            type="file"
                            accept=".pdf,.docx"
                            onChange={handleFileSelect}
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                          />
                          <FileText className={`h-12 w-12 mx-auto mb-4 ${isDragging ? 'text-primary' : 'text-muted-foreground'}`} />
                          <p className="font-medium mb-1">
                            Drag & drop resume (PDF, DOCX) or{' '}
                            <span className="text-primary underline cursor-pointer">browse</span>
                          </p>
                        </div>
                      )}

                      {uploadedFile && (
                        <div className="space-y-4">
                          <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/50 border">
                            <div className="flex items-center gap-3">
                              <FileText className="h-5 w-5 text-primary" />
                              <div>
                                <p className="font-medium text-sm">{uploadedFile.name}</p>
                                {parseMessage && (
                                  <p className="text-xs text-muted-foreground">{parseMessage}</p>
                                )}
                              </div>
                            </div>
                            <button onClick={clearFile} className="p-1 hover:bg-secondary rounded">
                              <X className="h-4 w-4" />
                            </button>
                          </div>

                          {parsing && (
                            <div className="flex items-center justify-center gap-2 py-4">
                              <Loader2 className="h-5 w-5 animate-spin text-primary" />
                              <span className="text-sm">Analyzing resume...</span>
                            </div>
                          )}

                          {!parsing && renderFormFields()}
                        </div>
                      )}

                      <p className="text-xs text-muted-foreground text-center mt-4">
                        Privacy First: Resumes processed instantly, never stored.
                      </p>
                    </TabsContent>

                    <TabsContent value="manual" className="mt-0">
                      {renderFormFields()}
                    </TabsContent>
                  </div>
                </Tabs>
              </CardContent>
            </Card>

            {/* Results Card */}
            <Card className="bg-gradient-card border-border/50">
              <CardHeader className="pb-2">
                <CardTitle className="text-xl flex items-center gap-2">
                  Prediction Results {result ? '' : '(Preview)'}
                  {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                </CardTitle>
              </CardHeader>
              <CardContent>
                {error && (
                  <div className="rounded-lg bg-red-500/10 border border-red-500/20 p-4 text-red-500 mb-4">
                    {error}
                  </div>
                )}

                {/* Bell Curve Visualization */}
                <div className="h-48 mb-4">
                  {bellCurveData ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={bellCurveData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <defs>
                          <linearGradient id="colorDensity" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="hsl(174, 72%, 56%)" stopOpacity={0.4}/>
                            <stop offset="95%" stopColor="hsl(174, 72%, 56%)" stopOpacity={0.1}/>
                          </linearGradient>
                        </defs>
                        <XAxis
                          dataKey="salary"
                          tickFormatter={(v) => `$${Math.round(v/1000)}k`}
                          stroke="hsl(200, 15%, 40%)"
                          fontSize={11}
                          tickLine={false}
                          axisLine={false}
                        />
                        <YAxis hide />
                        <ReferenceLine
                          x={result!.predicted_salary}
                          stroke="hsl(174, 72%, 56%)"
                          strokeWidth={2}
                          label={{
                            value: formatSalary(result!.predicted_salary),
                            position: 'top',
                            fill: 'hsl(174, 72%, 56%)',
                            fontSize: 14,
                            fontWeight: 'bold'
                          }}
                        />
                        <Area
                          type="monotone"
                          dataKey="density"
                          stroke="hsl(174, 72%, 56%)"
                          strokeWidth={2}
                          fill="url(#colorDensity)"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={generateBellCurve(100000, 145000, 190000)} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                          <defs>
                            <linearGradient id="colorDensityPreview" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="hsl(200, 15%, 40%)" stopOpacity={0.2}/>
                              <stop offset="95%" stopColor="hsl(200, 15%, 40%)" stopOpacity={0.05}/>
                            </linearGradient>
                          </defs>
                          <XAxis
                            dataKey="salary"
                            tickFormatter={(v) => `$${Math.round(v/1000)}k`}
                            stroke="hsl(200, 15%, 30%)"
                            fontSize={11}
                            tickLine={false}
                            axisLine={false}
                          />
                          <YAxis hide />
                          <ReferenceLine
                            x={145000}
                            stroke="hsl(200, 15%, 30%)"
                            strokeWidth={1}
                            strokeDasharray="3 3"
                            label={{
                              value: '90% Confidence Interval',
                              position: 'top',
                              fill: 'hsl(200, 15%, 40%)',
                              fontSize: 11
                            }}
                          />
                          <Area
                            type="monotone"
                            dataKey="density"
                            stroke="hsl(200, 15%, 30%)"
                            strokeWidth={1}
                            fill="url(#colorDensityPreview)"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>

                {/* X-Axis Labels */}
                <div className="flex justify-between text-xs px-4 mb-6">
                  <div className="text-center">
                    <span className="text-muted-foreground">Low</span>
                    {result && <p className="font-medium text-foreground">{formatSalary(result.salary_low)}</p>}
                  </div>
                  <div className="text-center">
                    <span className="text-muted-foreground">Median</span>
                    {result && <p className="font-medium text-primary">{formatSalary(result.predicted_salary)}</p>}
                  </div>
                  <div className="text-center">
                    <span className="text-muted-foreground">High</span>
                    {result && <p className="font-medium text-foreground">{formatSalary(result.salary_high)}</p>}
                  </div>
                </div>

                {/* Insights */}
                <div className="space-y-3">
                  {result && result.top_factors && result.top_factors.length > 0 ? (
                    // Dynamic factors from API
                    <>
                      {result.top_factors.map((factor, index) => (
                        <div key={index} className="flex items-start gap-3 text-sm">
                          {factor.impact === 'positive' ? (
                            <TrendingUp className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                          ) : (
                            <TrendingUp className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0 rotate-180" />
                          )}
                          <span>
                            <strong className={factor.impact === 'positive' ? 'text-green-500' : 'text-red-500'}>
                              {factor.impact === 'positive' ? '+' : '-'}{factor.name}:
                            </strong>{' '}
                            {factor.description}
                          </span>
                        </div>
                      ))}
                    </>
                  ) : (
                    // Placeholder when no result
                    <>
                      <div className="flex items-start gap-3 text-sm">
                        <TrendingUp className="h-4 w-4 text-muted-foreground mt-0.5 flex-shrink-0" />
                        <span className="text-muted-foreground">
                          <strong>Top Factors:</strong> [+Experience], [+Location], [-Missing Skill]
                        </span>
                      </div>
                      <div className="flex items-start gap-3 text-sm">
                        <MapPinned className="h-4 w-4 text-muted-foreground mt-0.5 flex-shrink-0" />
                        <span className="text-muted-foreground">
                          <strong>Location Insight:</strong> Compare to other tech hubs
                        </span>
                      </div>
                      <div className="flex items-start gap-3 text-sm">
                        <Sparkles className="h-4 w-4 text-muted-foreground mt-0.5 flex-shrink-0" />
                        <span className="text-muted-foreground">
                          <strong>Skill Gap:</strong> Add [Skill X] for potential increase
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Footer */}
          <footer className="text-center mt-12 text-sm text-muted-foreground">
            <p>
              Data Sources:{' '}
              <span className="text-foreground/70">H1B</span>,{' '}
              <span className="text-foreground/70">LinkedIn</span>,{' '}
              <span className="text-foreground/70">Adzuna</span>
            </p>
          </footer>
        </main>
      </div>
    </div>
  )

  function renderFormFields() {
    return (
      <div className="space-y-4">
        {/* Job Title */}
        <div className="space-y-2">
          <label className="text-sm font-medium flex items-center gap-2">
            <Briefcase className="h-4 w-4" />
            Job Title
          </label>
          <Select value={jobTitle} onChange={(e) => setJobTitle(e.target.value)}>
            {options?.job_titles.map((title) => (
              <option key={title} value={title}>{title}</option>
            )) || <option value="ML Engineer">ML Engineer</option>}
          </Select>
        </div>

        {/* Location */}
        <div className="space-y-2">
          <label className="text-sm font-medium flex items-center gap-2">
            <MapPin className="h-4 w-4" />
            Location
          </label>
          <Select value={location} onChange={(e) => setLocation(e.target.value)}>
            {options?.locations.map((loc) => (
              <option key={loc.code} value={loc.code}>{loc.name} ({loc.code})</option>
            )) || <option value="CA">California (CA)</option>}
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
            placeholder="e.g., Google, Meta, OpenAI"
          />
        </div>

        {/* Skills */}
        <div className="space-y-2">
          <label className="text-sm font-medium flex items-center gap-2">
            <Code className="h-4 w-4" />
            Skills
          </label>
          <div className="flex flex-wrap gap-2">
            {(options?.skills || ['Python', 'Machine Learning', 'Deep Learning', 'PyTorch', 'TensorFlow']).map((skill) => (
              <button
                key={skill}
                onClick={() => toggleSkill(skill)}
                className={`px-3 py-1 text-sm rounded-full border transition-colors ${
                  selectedSkills.includes(skill)
                    ? 'bg-primary text-primary-foreground border-primary'
                    : 'bg-background border-input hover:bg-accent hover:text-accent-foreground'
                }`}
              >
                {skill}
              </button>
            ))}
          </div>
        </div>
      </div>
    )
  }
}
