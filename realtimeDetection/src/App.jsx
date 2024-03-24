// Home.jsx
import './index.css'; // Import the CSS file
import Home from './Home.jsx'; // Import the Home
import img from './res-img.png';

const App = () => {
  const scrollToCanvas = () => {
    const canvasElement = document.getElementById('canvas');
    canvasElement.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <>
      <header>
        <nav>
          <h1 className="logo">TextDetector</h1>
        </nav>
      </header>
      <main>
        <div className="container">
          <img src={img} alt="" />
          <div className="hero-text">
            <h1>Your favorite object text detector</h1>
            <p>Open up the detector and watch it detect objects and print out the text is sees in realtime.</p>
            <button onClick={scrollToCanvas}>Try now</button>
          </div>
        </div>
      </main>
      <Home /> {/* Render the Canvas component */}
    </>
  );
}

export default App;
