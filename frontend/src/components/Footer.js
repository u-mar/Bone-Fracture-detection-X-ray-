import { Link } from "react-router-dom";
import "../Style/footer.css";

export default function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-content">
          {/* Company Info */}
          <div className="footer-section">
            <div className="footer-logo">
              <span className="footer-logo-icon">🏥</span>
              <span className="footer-logo-text">MedScan</span>
            </div>
            <p className="footer-description">
              AI-powered medical image analysis for accurate bone fracture detection. 
              Empowering healthcare with cutting-edge technology.
            </p>
            <div className="social-links">
              <a href="#" className="social-link" aria-label="Facebook">
                <i className="social-icon">f</i>
              </a>
              <a href="#" className="social-link" aria-label="Twitter">
                <i className="social-icon">𝕏</i>
              </a>
              <a href="#" className="social-link" aria-label="LinkedIn">
                <i className="social-icon">in</i>
              </a>
              <a href="#" className="social-link" aria-label="GitHub">
                <i className="social-icon">⚙</i>
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div className="footer-section">
            <h3 className="footer-heading">Quick Links</h3>
            <ul className="footer-links">
              <li><Link to="/">Home</Link></li>
              <li><Link to="/detect">Detection</Link></li>
              <li><a href="#features">Features</a></li>
              <li><a href="#about">About Us</a></li>
            </ul>
          </div>

          {/* Resources */}
          <div className="footer-section">
            <h3 className="footer-heading">Resources</h3>
            <ul className="footer-links">
              <li><a href="#docs">Documentation</a></li>
              <li><a href="#api">API Reference</a></li>
              <li><a href="#support">Support</a></li>
              <li><a href="#faq">FAQ</a></li>
            </ul>
          </div>

          {/* Contact */}
          <div className="footer-section">
            <h3 className="footer-heading">Contact Us</h3>
            <ul className="footer-contact">
              <li>
                <span className="contact-icon">📧</span>
                <span>support@medscan.com</span>
              </li>
              <li>
                <span className="contact-icon">📞</span>
                <span>+1 (555) 123-4567</span>
              </li>
              <li>
                <span className="contact-icon">📍</span>
                <span>123 Medical Plaza, Healthcare City</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Footer Bottom */}
        <div className="footer-bottom">
          <p className="footer-copyright">
            © {currentYear} MedScan. All rights reserved.
          </p>
          <div className="footer-legal">
            <a href="#privacy">Privacy Policy</a>
            <span className="separator">•</span>
            <a href="#terms">Terms of Service</a>
            <span className="separator">•</span>
            <a href="#cookies">Cookie Policy</a>
          </div>
        </div>
      </div>
    </footer>
  );
}
