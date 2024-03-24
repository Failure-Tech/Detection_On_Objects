import axios from "axios";
import { useState } from "react";

function Home() {
  const [camera, setCamera] = useState(false);
  const [text, setText] = useState(null);

  const openDetector = async () => {
    try {
      // Clear the text when starting the detection
      setText(null);

      const response = await axios.post('http://localhost:5000/start_detection');
      setCamera(response.data.message);
      const detectedText = response.data.texts;
      console.log("Object Detection working Successfully");
      console.log(detectedText);
      setText(detectedText.join(' ')); // Join the detected text with spaces
    } catch (error) {
      console.error('Error starting detection:', error);
      setCamera('Error starting detection');
    }
  }

  return (
    <>
      <button onClick={openDetector}>Open Detector</button>
      <p>{camera}</p>

      <div className="output">
        <h1>Output</h1>
        {text !== null && (
          <div>Predicted Text: {text}</div>
        )}
      </div>
    </>
  )
}

export default Home;
