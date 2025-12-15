import { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../Style/boneStyle.css";

export default function BoneFractureDetection() {
  const [image, setImage] = useState(null);
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [imagePreview, setImagePreview] = useState(null);
  const navigate = useNavigate();

  const handleImageChange = (e) => {
    const selectedImage = e.target.files[0];
    setImage(selectedImage);
    setImagePreview(URL.createObjectURL(selectedImage));
  };

  const handleSubmit = async () => {
    if (!image || !name || age === "" || Number(age) < 0) {
      alert("Please fill in all fields with valid data. Age must be non-negative.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);
    formData.append("name", name);
    formData.append("age", age);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      console.log("submit", data);
      // Pass the originalImage to the result object
      const result = {
        ...data,
        originalImage: data.originalImage
      };
      navigate("/report", { state: { name, age, result } });
    } catch (error) {
      console.error("Error uploading image", error);
    }
  };

  return (
    <div className="container classic-bg">
      <div className="card glass-effect slide-in">
        <h2 className="title gradient-text">
          🦴 MedScan: AI Powered Medical Image Segmentation
        </h2>
        <div className="form-container">
          <div className="form">
            <div className="input-group">
              <label htmlFor="name">👤 Patient Name</label>
              <input
                id="name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="input"
              />
            </div>
            <div className="input-group">
              <label htmlFor="age">🎂 Age</label>
              <input
                id="age"
                type="number"
                min="0"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                className="input"
              />
            </div>
            <div className="input-group">
              <label htmlFor="file">📷 Upload X-ray Image</label>
              <input
                id="file"
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                className="input"
              />
            </div>
            <button onClick={handleSubmit} className="btn neon-btn">
              🚀 Analyze
            </button>
          </div>

          <div className="image-preview-container fade-in">
            {imagePreview ? (
              <>
                <h3 className="gradient-text">🖼️ Uploaded X-ray</h3>
                <img
                  src={imagePreview}
                  alt="Uploaded X-ray"
                  className="uploaded-image"
                />
              </>
            ) : (
              <p className="placeholder-text">No image uploaded</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}