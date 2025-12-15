import { Link } from "react-router-dom";
import "../Style/landing.css";

export default function LandingPage() {
  return (
    <div className="landing-page">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <div className="hero-text">
            <h1 className="hero-title">
              AI-Powered Medical <span className="highlight">Image Analysis</span>
            </h1>
            <p className="hero-description">
              Harness the power of artificial intelligence to detect bone fractures 
              and analyze medical images with unprecedented accuracy. Fast, reliable, 
              and accessible healthcare technology at your fingertips.
            </p>
            <div className="hero-buttons">
              <Link to="/detect" className="btn-primary">
                Start Detection
              </Link>
              <a href="#features" className="btn-secondary">
                Learn More
              </a>
            </div>
          </div>
          <div className="hero-image">
            <div className="floating-card">
              <div className="card-icon">🔬</div>
              <div className="card-text">AI Analysis</div>
            </div>
            <div className="floating-card delay-1">
              <div className="card-icon">🦴</div>
              <div className="card-text">Bone Detection</div>
            </div>
            <div className="floating-card delay-2">
              <div className="card-icon">📊</div>
              <div className="card-text">Detailed Reports</div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="features-section">
        <div className="section-header">
          <h2 className="section-title">Why Choose MedScan?</h2>
          <p className="section-subtitle">
            Advanced technology meets medical expertise
          </p>
        </div>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🎯</div>
            <h3 className="feature-title">High Accuracy</h3>
            <p className="feature-description">
              Our AI models are trained on thousands of medical images to provide 
              accurate fracture detection with minimal false positives.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h3 className="feature-title">Fast Results</h3>
            <p className="feature-description">
              Get instant analysis results within seconds. No more waiting for 
              hours or days to receive your medical image reports.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🔒</div>
            <h3 className="feature-title">Secure & Private</h3>
            <p className="feature-description">
              Your medical data is encrypted and protected. We prioritize patient 
              privacy and comply with healthcare data standards.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📱</div>
            <h3 className="feature-title">Easy to Use</h3>
            <p className="feature-description">
              Simple, intuitive interface designed for both medical professionals 
              and patients. Upload, analyze, and download reports effortlessly.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🧠</div>
            <h3 className="feature-title">Deep Learning</h3>
            <p className="feature-description">
              Powered by state-of-the-art neural networks that continuously learn 
              and improve from new medical imaging data.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📄</div>
            <h3 className="feature-title">Detailed Reports</h3>
            <p className="feature-description">
              Comprehensive PDF reports with visual annotations, confidence scores, 
              and recommendations for next steps.
            </p>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="how-it-works-section">
        <div className="section-header">
          <h2 className="section-title">How It Works</h2>
          <p className="section-subtitle">
            Simple process, powerful results
          </p>
        </div>
        <div className="steps-container">
          <div className="step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h3 className="step-title">Upload Image</h3>
              <p className="step-description">
                Upload your X-ray image along with patient information
              </p>
            </div>
          </div>
          <div className="step-arrow">→</div>
          <div className="step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h3 className="step-title">AI Analysis</h3>
              <p className="step-description">
                Our AI model processes and analyzes the medical image
              </p>
            </div>
          </div>
          <div className="step-arrow">→</div>
          <div className="step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h3 className="step-title">Get Results</h3>
              <p className="step-description">
                Receive detailed report with fracture detection results
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="stats-section">
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-number">95%</div>
            <div className="stat-label">Accuracy Rate</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">10K+</div>
            <div className="stat-label">Images Analyzed</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">&lt;5s</div>
            <div className="stat-label">Average Processing Time</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">24/7</div>
            <div className="stat-label">Available</div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="cta-content">
          <h2 className="cta-title">Ready to Get Started?</h2>
          <p className="cta-description">
            Experience the future of medical image analysis today
          </p>
          <Link to="/detect" className="cta-button">
            Start Detection Now
          </Link>
        </div>
      </section>
    </div>
  );
}
