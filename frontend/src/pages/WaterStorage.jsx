import { useState, useEffect } from 'react';
import { usePredictionContext } from '../context/PredictionContext';
import { calculateStorage, predictRainfall } from '../api/apiClient';
import StorageChart from '../components/StorageChart';
import { fetchSamplePredictions } from '../api/apiClient';

export default function WaterStorage() {
  const { predictions: storedPreds, clearPredictions } = usePredictionContext();
  const [catchmentArea, setCatchmentArea] = useState(10000);
  const [runoffCoef, setRunoffCoef] = useState(0.75);
  const [resEff, setResEff] = useState(0.85);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    if (storedPreds && storedPreds.length > 0) {
      handleCalculate(storedPreds);
    } else {
      // Auto-load sample if no predictions exist
      fetchSamplePredictions().then(res => handleCalculate(res.predictions || []));
    }
  }, []);

  const handleCalculate = async (preds) => {
    if (!preds || preds.length === 0) return;
    setLoading(true);
    try {
      const res = await calculateStorage({
        predictions: preds,
        catchment_area_m2: parseFloat(catchmentArea),
        runoff_coefficient: parseFloat(runoffCoef),
        reservoir_efficiency: parseFloat(resEff)
      });
      setResults(res.daily_breakdown);
      setSummary(res.summary);
    } catch (err) {
      console.error('Storage calc failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (storedPreds) handleCalculate(storedPreds);
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Water Storage Calculator</h1>
      <form onSubmit={handleSubmit} className="bg-dark-card p-4 rounded-xl border border-dark-border grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
        <div>
          <label className="block text-sm text-dark-muted mb-1">Catchment Area (m²)</label>
          <input type="number" value={catchmentArea} onChange={e => setCatchmentArea(e.target.value)} className="w-full bg-slate-800 border border-dark-border rounded-lg px-3 py-2 text-white" />
        </div>
        <div>
          <label className="block text-sm text-dark-muted mb-1">Runoff Coefficient ({runoffCoef})</label>
          <input type="range" min="0.1" max="1.0" step="0.05" value={runoffCoef} onChange={e => setRunoffCoef(e.target.value)} className="w-full accent-brand-blue" />
        </div>
        <div>
          <label className="block text-sm text-dark-muted mb-1">Reservoir Efficiency ({resEff})</label>
          <input type="range" min="0.1" max="1.0" step="0.05" value={resEff} onChange={e => setResEff(e.target.value)} className="w-full accent-brand-teal" />
        </div>
        <button type="submit" disabled={loading} className="bg-brand-teal hover:bg-brand-teal/80 text-white px-4 py-2 rounded-lg font-medium transition disabled:opacity-50">
          {loading ? '⏳ Calculating...' : '💧 Calculate Storage'}
        </button>
      </form>

      {results && results.length > 0 && (
        <>
          <StorageChart data={results} />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-dark-card p-4 rounded-xl border border-dark-border">
              <h3 className="text-lg font-semibold mb-2">Storage Summary</h3>
              <div className="space-y-2 text-sm">
                <p>Total Predicted Rainfall: <span className="font-bold text-brand-blue">{summary.total_predicted_rainfall_mm} mm</span></p>
                <p>Water Collected: <span className="font-bold text-brand-teal">{summary.total_water_collected_L.toLocaleString()} L</span></p>
                <p>Evaporation Loss: <span className="font-bold text-brand-coral">{summary.total_evaporation_loss_L.toLocaleString()} L</span></p>
                <p>Net Harvestable: <span className="font-bold text-brand-green">{summary.net_harvestable_L.toLocaleString()} L</span></p>
              </div>
            </div>
            <div className="bg-dark-card rounded-xl border border-dark-border overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="bg-slate-800/50 text-dark-muted">
                  <tr>
                    <th className="px-3 py-2">Date</th>
                    <th className="px-3 py-2">Rain (mm)</th>
                    <th className="px-3 py-2">Collected (L)</th>
                    <th className="px-3 py-2">Loss (L)</th>
                    <th className="px-3 py-2">Net (L)</th>
                    <th className="px-3 py-2">Cumulative (L)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-dark-border">
                  {results.map((r, i) => (
                    <tr key={i} className="hover:bg-slate-700/20">
                      <td className="px-3 py-1">{new Date(r.date).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' })}</td>
                      <td className="px-3 py-1">{r.predicted_rainfall_mm}</td>
                      <td className="px-3 py-1">{r.water_collected_L.toLocaleString()}</td>
                      <td className="px-3 py-1">{r.evaporation_loss_L.toLocaleString()}</td>
                      <td className="px-3 py-1">{r.net_storage_L.toLocaleString()}</td>
                      <td className="px-3 py-1 font-bold">{r.cumulative_storage_L.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}