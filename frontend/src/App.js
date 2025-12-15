import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Login from "./components/Login";
import LandingPage from "./components/LandingPage";
import BoneFractureDetection from "./components/BoneFractureDetection";
import ReportPage from "./components/Report";
import "./App.css";

function App() {
  return (
    <Router>
      <Routes>
        {/* Login page without Navbar/Footer */}
        <Route path="/" element={<Login />} />
        
        {/* Main app routes with Navbar/Footer */}
        <Route
          path="/*"
          element={
            <div className="app-wrapper">
              <Navbar />
              <main className="main-content">
                <Routes>
                  <Route path="/home" element={<LandingPage />} />
                  <Route path="/detect" element={<BoneFractureDetection />} />
                  <Route path="/report" element={<ReportPage/>} />
                </Routes>
              </main>
              <Footer />
            </div>
          }
        />
      </Routes>
    </Router>
  );
}

export default App;
