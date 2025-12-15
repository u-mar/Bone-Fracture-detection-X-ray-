import { Link, useLocation } from "react-router-dom";
import "../Style/navbar.css";

export default function Navbar() {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/home" className="navbar-logo">
          <span className="logo-icon">🏥</span>
          <span className="logo-text">MedScan</span>
        </Link>
        
        <div className="navbar-menu">
          <Link 
            to="/home" 
            className={`nav-link ${location.pathname === "/home" ? "active" : ""}`}
          >
            Home
          </Link>
          <Link 
            to="/detect" 
            className={`nav-link ${location.pathname === "/detect" ? "active" : ""}`}
          >
            Detection
          </Link>
          <Link to="/detect" className="nav-btn">
            Get Started
          </Link>
        </div>
      </div>
    </nav>
  );
}
