import { useState, useEffect, useCallback } from 'react'
import { DollarSign, Briefcase, MapPin, Clock, Building2, Code, Loader2, TrendingUp, TrendingDown, Github, Upload, FileText, X, ToggleLeft, ToggleRight } from 'lucide-react'
import { Input } from './components/ui/input'
import { Select } from './components/ui/select'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from './components/ui/card'
import { Slider } from './components/ui/slider'

// Use relative /api path in production (same origin), fallback to localhost for dev
const API_URL = import.meta.env.VITE_API_URL || (window.location.hostname === 'localhost' ? 'http://localhost:8000' : '/api')

interface Options {
  job_titles: string[]
  locations: { code: string; name: string }[]
  skills: string[]
}

interface PredictionResult {
  predicted_salary: number
  salary_low: number
  salary_high: number
  confidence_level: string
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

export default function App() {
  const [options, setOptions] = useState<Options | null>(null)
  const [jobTitle, setJobTitle] = useState('ML Engineer')
  const [location, setLocation] = useState('CA')
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
      .then(res => res.json())
      .then(data => setOptions(data))
      .catch(err => console.error('Failed to fetch options:', err))
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
      setError(err instanceof Error ? err.message : 'Failed to get prediction')
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
      setError(err instanceof Error ? err.message : 'Failed to parse resume')
    } finally {
      setParsing(false)
    }
  }

  const clearFile = () => {
    setUploadedFile(null)
    setParseMessage(null)
    setResult(null)
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
            Upload your resume or enter details manually for instant salary estimates
          </p>
        </div>

        {/* Mode Toggle */}
        <div className="flex justify-center mb-6">
          <button
            onClick={() => setInputMode(inputMode === 'upload' ? 'manual' : 'upload')}
            className="flex items-center gap-2 px-4 py-2 rounded-full border bg-secondary/50 hover:bg-secondary transition-colors"
          >
            {inputMode === 'upload' ? (
              <>
                <ToggleLeft className="h-5 w-5" />
                <span className="text-sm font-medium">Switch to Manual Input</span>
              </>
            ) : (
              <>
                <ToggleRight className="h-5 w-5" />
                <span className="text-sm font-medium">Switch to Resume Upload</span>
              </>
            )}
          </button>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          {/* Input Form */}
          <Card>
            <CardHeader>
              <CardTitle className="text-xl">
                {inputMode === 'upload' ? 'Upload Resume' : 'Job Details'}
              </CardTitle>
              <CardDescription>
                {inputMode === 'upload'
                  ? 'Drop your resume to auto-fill job details'
                  : 'Predictions update automatically as you type'}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Resume Upload Zone */}
              {inputMode === 'upload' && !uploadedFile && (
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                    isDragging
                      ? 'border-primary bg-primary/5'
                      : 'border-muted-foreground/25 hover:border-primary/50'
                  }`}
                >
                  <input
                    type="file"
                    accept=".pdf,.docx"
                    onChange={handleFileSelect}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />
                  <Upload className={`h-10 w-10 mx-auto mb-4 ${isDragging ? 'text-primary' : 'text-muted-foreground'}`} />
                  <p className="font-medium mb-1">
                    {isDragging ? 'Drop your resume here' : 'Drag & drop your resume'}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    or click to browse (PDF, DOCX)
                  </p>
                </div>
              )}

              {/* Uploaded File Status */}
              {uploadedFile && (
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
                  <button
                    onClick={clearFile}
                    className="p-1 hover:bg-secondary rounded"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              )}

              {/* Parsing Indicator */}
              {parsing && (
                <div className="flex items-center justify-center gap-2 py-4">
                  <Loader2 className="h-5 w-5 animate-spin text-primary" />
                  <span className="text-sm">Analyzing resume...</span>
                </div>
              )}

              {/* Form Fields (shown in manual mode or after upload) */}
              {(inputMode === 'manual' || uploadedFile) && !parsing && (
                <>
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
                              : 'bg-background border-input hover:bg-accent'
                          }`}
                        >
                          {skill}
                        </button>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>

          {/* Results */}
          <Card>
            <CardHeader>
              <CardTitle className="text-xl flex items-center gap-2">
                Prediction Results
                {loading && <Loader2 className="h-4 w-4 animate-spin" />}
              </CardTitle>
              <CardDescription>
                {result ? 'Salary estimate based on your profile' : 'Results update in real-time'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {error && (
                <div className="rounded-lg bg-red-500/10 border border-red-500/20 p-4 text-red-500">
                  {error}
                </div>
              )}

              {!result && !error && (
                <div className="text-center py-12 text-muted-foreground">
                  {inputMode === 'upload' && !uploadedFile
                    ? 'Upload a resume to see salary predictions'
                    : 'Enter job details to see predictions'}
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
                    {selectedSkills.length > 0 && (
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Skills</span>
                        <span className="font-medium text-right">{selectedSkills.join(', ')}</span>
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
