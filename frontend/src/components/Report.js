import { useLocation, useNavigate } from "react-router-dom";
import "../Style/reportStyle.css";
import jsPDF from 'jspdf';

export default function ReportPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { name, age, result } = location.state || {};
  const currentDate = new Date().toLocaleDateString();
  const currentTime = new Date().toLocaleTimeString();

  if (!result) {
    return (
      <div className="report-container classic-bg">
        <div className="report-card glass-effect slide-in">
          <h2 className="gradient-text">📄 No Report Found</h2>
          <button onClick={() => navigate("/")} className="btn golden-btn">
            ⬅ Go Back
          </button>
        </div>
      </div>
    );
  }

  // Download Report as PDF
  const handleDownloadReport = () => {
    const pdf = new jsPDF();
    
    // Color scheme
    const primaryColor = [102, 126, 234]; // Purple
    const secondaryColor = [118, 75, 162]; // Dark purple
    const accentColor = [255, 215, 0]; // Gold
    const textColor = [45, 55, 72]; // Dark gray
    const lightBg = [248, 249, 250]; // Light gray
    
    // Header with gradient effect
    pdf.setFillColor(primaryColor[0], primaryColor[1], primaryColor[2]);
    pdf.rect(0, 0, 210, 45, 'F');
    
    // Logo/Icon (Medical cross)
    pdf.setFillColor(255, 255, 255);
    pdf.rect(15, 12, 3, 20, 'F'); // Vertical bar
    pdf.rect(7, 20, 19, 3, 'F'); // Horizontal bar
    
    // Title
    pdf.setTextColor(255, 255, 255);
    pdf.setFontSize(24);
    pdf.setFont("helvetica", "bold");
    pdf.text("MedScan AI", 35, 22);
    pdf.setFontSize(14);
    pdf.setFont("helvetica", "normal");
    pdf.text("Bone Fracture Analysis Report", 35, 32);
    
    // Reset text color
    pdf.setTextColor(textColor[0], textColor[1], textColor[2]);
    
    // Patient Information Section
    let yPos = 55;
    
    // Section header
    pdf.setFillColor(lightBg[0], lightBg[1], lightBg[2]);
    pdf.roundedRect(15, yPos - 5, 180, 10, 2, 2, 'F');
    pdf.setFontSize(16);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
    pdf.text("Patient Information", 20, yPos + 2);
    
    // Patient details box
    yPos += 15;
    pdf.setDrawColor(primaryColor[0], primaryColor[1], primaryColor[2]);
    pdf.setLineWidth(0.5);
    pdf.roundedRect(15, yPos, 180, 35, 3, 3);
    
    pdf.setFontSize(12);
    pdf.setFont("helvetica", "normal");
    pdf.setTextColor(textColor[0], textColor[1], textColor[2]);
    
    yPos += 10;
    pdf.setFont("helvetica", "bold");
    pdf.text("Name:", 20, yPos);
    pdf.setFont("helvetica", "normal");
    pdf.text(name, 50, yPos);
    
    yPos += 8;
    pdf.setFont("helvetica", "bold");
    pdf.text("Age:", 20, yPos);
    pdf.setFont("helvetica", "normal");
    pdf.text(age.toString(), 50, yPos);
    
    yPos += 8;
    pdf.setFont("helvetica", "bold");
    pdf.text("Date:", 105, yPos - 16);
    pdf.setFont("helvetica", "normal");
    pdf.text(currentDate, 130, yPos - 16);
    
    pdf.setFont("helvetica", "bold");
    pdf.text("Time:", 105, yPos - 8);
    pdf.setFont("helvetica", "normal");
    pdf.text(currentTime, 130, yPos - 8);
    
    // Analysis Results Section
    yPos += 15;
    
    // Section header
    pdf.setFillColor(lightBg[0], lightBg[1], lightBg[2]);
    pdf.roundedRect(15, yPos - 5, 180, 10, 2, 2, 'F');
    pdf.setFontSize(16);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(primaryColor[0], primaryColor[1], primaryColor[2]);
    pdf.text("Analysis Results", 20, yPos + 2);
    
    yPos += 15;
    
    // Fracture Status with color coding
    pdf.setFontSize(13);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(textColor[0], textColor[1], textColor[2]);
    pdf.text("Fracture Status:", 20, yPos);
    
    if (result.fracture_detected) {
      pdf.setTextColor(220, 53, 69); // Red for fracture
      pdf.setFont("helvetica", "bold");
      pdf.text("FRACTURE DETECTED", 70, yPos);
    } else {
      pdf.setTextColor(40, 167, 69); // Green for no fracture
      pdf.setFont("helvetica", "bold");
      pdf.text("NO FRACTURE DETECTED", 70, yPos);
    }
    
    pdf.setTextColor(textColor[0], textColor[1], textColor[2]);
    
    if (result.fracture_detected) {
      yPos += 12;
      pdf.setFont("helvetica", "bold");
      pdf.text("Location:", 20, yPos);
      pdf.setFont("helvetica", "normal");
      pdf.text(result.location || "Not specified", 70, yPos);
      
      yPos += 12;
      pdf.setFont("helvetica", "bold");
      pdf.text("Severity:", 20, yPos);
      
      // Color-code severity
      const severity = result.severity || "Unknown";
      if (severity.toLowerCase() === "severe") {
        pdf.setTextColor(220, 53, 69); // Red
      } else if (severity.toLowerCase() === "moderate") {
        pdf.setTextColor(255, 193, 7); // Yellow/Orange
      } else if (severity.toLowerCase() === "mild") {
        pdf.setTextColor(40, 167, 69); // Green
      }
      pdf.setFont("helvetica", "bold");
      pdf.text(severity.toUpperCase(), 70, yPos);
      pdf.setTextColor(textColor[0], textColor[1], textColor[2]);
    }
    
    // Images Section
    if (result.originalImage) {
      pdf.addPage();
      
      // Page header
      pdf.setFillColor(primaryColor[0], primaryColor[1], primaryColor[2]);
      pdf.rect(0, 0, 210, 30, 'F');
      pdf.setTextColor(255, 255, 255);
      pdf.setFontSize(18);
      pdf.setFont("helvetica", "bold");
      pdf.text("Original X-ray Image", 20, 18);
      
      try {
        pdf.addImage(
          `data:image/png;base64,${result.originalImage}`,
          "PNG",
          15,
          40,
          180,
          180
        );
      } catch (error) {
        console.error("Error adding original image:", error);
      }
    }

    if (result.heatmap) {
      pdf.addPage();
      
      // Page header
      pdf.setFillColor(secondaryColor[0], secondaryColor[1], secondaryColor[2]);
      pdf.rect(0, 0, 210, 30, 'F');
      pdf.setTextColor(255, 255, 255);
      pdf.setFontSize(18);
      pdf.setFont("helvetica", "bold");
      pdf.text("Analysis Visualization", 20, 18);
      
      try {
        pdf.addImage(
          `data:image/png;base64,${result.heatmap}`,
          "PNG",
          15,
          40,
          180,
          180
        );
      } catch (error) {
        console.error("Error adding heatmap:", error);
      }
    }

    // Disclaimer Section
    pdf.addPage();
    
    // Page header
    pdf.setFillColor(accentColor[0], accentColor[1], accentColor[2]);
    pdf.rect(0, 0, 210, 30, 'F');
    pdf.setTextColor(textColor[0], textColor[1], textColor[2]);
    pdf.setFontSize(18);
    pdf.setFont("helvetica", "bold");
    pdf.text("Important Medical Disclaimer", 20, 18);
    
    yPos = 50;
    pdf.setFontSize(11);
    pdf.setFont("helvetica", "normal");
    pdf.setTextColor(textColor[0], textColor[1], textColor[2]);
    
    const disclaimerText = [
      "This report is generated by AI-assisted analysis and is intended for informational purposes only.",
      "",
      "IMPORTANT:",
      "• This analysis should NOT replace professional medical diagnosis",
      "• Always consult with a qualified physician or radiologist",
      "• AI analysis may have limitations and errors",
      "• Final diagnosis must be made by licensed medical professionals",
      "",
      "For any medical concerns or emergency, please seek immediate professional medical attention."
    ];
    
    disclaimerText.forEach((line) => {
      if (line.startsWith("•")) {
        pdf.text(line, 25, yPos);
      } else if (line === "IMPORTANT:") {
        pdf.setFont("helvetica", "bold");
        pdf.text(line, 20, yPos);
        pdf.setFont("helvetica", "normal");
      } else {
        pdf.text(line, 20, yPos);
      }
      yPos += 7;
    });
    
    // Footer on last page
    yPos = 270;
    pdf.setDrawColor(primaryColor[0], primaryColor[1], primaryColor[2]);
    pdf.setLineWidth(0.5);
    pdf.line(20, yPos, 190, yPos);
    
    yPos += 8;
    pdf.setFontSize(9);
    pdf.setTextColor(128, 128, 128);
    pdf.text("Generated by MedScan AI | AI-Powered Medical Image Analysis", 20, yPos);
    pdf.text(`Report Date: ${currentDate} ${currentTime}`, 140, yPos);

    // Save the PDF
    pdf.save(`${name}_MedScan_Report_${currentDate.replace(/\//g, '-')}.pdf`);
  };

  // Download Heatmap Image
  const handleDownloadHeatmap = () => {
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${result.heatmap}`;
    link.download = `${name}_heatmap.png`;
    link.click();
  };

  return (
    <div className="report-container classic-bg">
      <div className="report-card glass-effect slide-in">
        <h2 className="report-title gradient-text">🦴 Bone Fracture Detection Report</h2>

        <div className="report-details fade-in">
          <div className="patient-info">
            <h3 className="section-title">Patient Information</h3>
            <p><strong>👤 Name:</strong> {name}</p>
            <p><strong>🎂 Age:</strong> {age}</p>
            <p><strong>📅 Date:</strong> {currentDate}</p>
            <p><strong>🕒 Time:</strong> {currentTime}</p>
          </div>

            <div className="analysis-results">
            <h3 className="section-title">Analysis Results</h3>
            <p>
              <strong>🩻 Fracture Status:</strong>{" "}
              <span className={!result.fracture_detected ? "no-fracture" : "fracture"}>
                {!result.fracture_detected ? "No fracture detected" : "Fracture detected"}
              </span>
            </p>
            {result.fracture_detected && (
              <>
                <p>
                  <strong>📍 Location:</strong>{" "}
                  <span className="highlight-text">{result.location || "Not specified"}</span>
                </p>
                <p>
                  <strong>⚠️ Severity:</strong>{" "}
                  <span className={`severity-${(result.severity || "").toLowerCase()}`}>
                    {result.severity || "Assessment required"}
                  </span>
                </p>
              </>
            )}
          </div>

          <div className="original-image-section">
            <h3 className="section-title">Original X-ray Image</h3>
            <div className="image-container">
              {result && result.originalImage ? (
                <img
                  src={`data:image/png;base64,${result.originalImage}`}
                  alt="Original X-ray"
                  className="xray-image"
                  onError={(e) => {
                    console.error("Error loading image");
                    e.target.parentElement.innerHTML = '<p class="no-image-text">Error loading image</p>';
                  }}
                />
              ) : (
                <p className="no-image-text">No X-ray image available</p>
              )}
            </div>
          </div>
          
          {result && result.heatmap && (
            <div className="heatmap-section">
              <h3 className="section-title">Analysis Visualization</h3>
              <div className="image-container">
                <img
                  src={`data:image/png;base64,${result.heatmap}`}
                  alt="Analysis Visualization"
                  className="heatmap-image"
                  onError={(e) => {
                    console.error("Error loading heatmap");
                    e.target.parentElement.innerHTML = '<p class="no-image-text">Error loading visualization</p>';
                  }}
                />
              </div>
            </div>
          )}
        </div>

        {result.fracture && (
          <div className="heatmap-section fade-in">
            <h3 className="gradient-text">🔥 Fracture Heatmap</h3>
            <img
              src={`data:image/png;base64,${result.heatmap}`}
              alt="Fracture Heatmap"
              className="heatmap"
            />
            <button onClick={handleDownloadHeatmap} className="btn neon-btn mt-2">
              📥 Download Heatmap
            </button>
          </div>
        )}

        <div className="button-group">
          <button onClick={handleDownloadReport} className="btn golden-btn">
            📄 Download Report
          </button>
          <button onClick={() => navigate("/")} className="btn neon-btn">
            🔁 New Report
          </button>
        </div>
      </div>
    </div>
  );
}
