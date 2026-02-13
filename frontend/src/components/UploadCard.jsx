import React, { useState } from "react";
import axios from "axios";

function UploadCard() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFile = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult("");
    setConfidence(null);
  };

  const handlePredict = async () => {
    if (!image) {
      alert("Please upload an image");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      setLoading(true);

      const res = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          timeout: 10000
        }
      );

      console.log(res.data);

      setResult(res.data.prediction);
      setConfidence(res.data.confidence);

    } catch (error) {
      console.error(error);
      alert("Error connecting to server. Check backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-card">
      <h2>AI Breast Cancer Analyzer</h2>
      <p>Upload histopathological image</p>

      <label className="upload-box">
        <input type="file" accept="image/*" onChange={handleFile} hidden />
        {preview ? (
          <img src={preview} alt="Preview" className="preview" />
        ) : (
          <span>📁 Click to Upload Image</span>
        )}
      </label>

      <button className="predict-btn" onClick={handlePredict}>
        {loading ? "Analyzing..." : "Analyze Image"}
      </button>

      {result && (
        <div className={`result ${result === "Malignant" ? "danger" : "safe"}`}>
          <p>{result}</p>
          {confidence !== null && (
            <p>Confidence: <b>{confidence}%</b></p>
          )}
        </div>
      )}
    </div>
  );
}

export default UploadCard;
