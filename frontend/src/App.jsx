import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { PredictionProvider } from './context/PredictionContext';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import RainfallPredictor from './pages/RainfallPredictor';
import WaterStorage from './pages/WaterStorage';
import IrrigationScheduler from './pages/IrrigationScheduler';
import ModelComparison from './pages/ModelComparison';

export default function App() {
  return (
    <PredictionProvider>
      <Router>
        <div className="min-h-screen bg-dark-bg">
          <Navbar />
          <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predict" element={<RainfallPredictor />} />
              <Route path="/storage" element={<WaterStorage />} />
              <Route path="/irrigation" element={<IrrigationScheduler />} />
              <Route path="/compare" element={<ModelComparison />} />
            </Routes>
          </main>
          <footer className="text-center text-dark-muted py-6 text-sm border-t border-dark-border mt-12">
            © 2024 DK-AquaPredict • Smart Water Management for Dakshina Kannada
          </footer>
        </div>
      </Router>
    </PredictionProvider>
  );
}