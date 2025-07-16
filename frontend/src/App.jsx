import React, { useState } from "react";
import UploadForm from "./components/UploadForm";

function App() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  return (
    <div className="app-container">
      <h1>License Plate Detection</h1>
      <UploadForm setResult={setResult} setError={setError} />
      {error && <p className="error">{error}</p>}
      {result && (
        <div className="result">
          <h2>Detected License Plate:</h2>
          <p>{result}</p>
        </div>
      )}
    </div>
  );
}

export default App;
