import { useState } from 'react';
import { predictRainfall } from '../api/apiClient';
import { usePredictionContext } from '../context/PredictionContext';
import RainfallChart from '../components/RainfallChart';

const MODELS = [
  { value: 'lstm', label: 'LSTM' },
  { value: 'bilstm', label: 'Bidirectional LSTM' },
  { value: 'gru', label: 'GRU' },
  { value: 'cnn_lstm', label: 'CNN-LSTM Hybrid' },
  { value: 'transformer', label: 'Transformer' },
  { value: 'autoencoder', label: 'Autoencoder + Regression' },
];

export default function RainfallPredictor() {
  const [startDate, setStartDate] = useState(new Date().toISOString().split('T')[0]);
  const [days, setDays] = useState(30);
  const [model, setModel] = useState('lstm');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const { storePredictions } = usePredictionContext();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await predictRainfall({ start_date: startDate, days: parseInt(days), model });
      setPredictions(data);
      storePredictions(data, { model, days, startDate });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Rainfall Predictor</h1>
      <form onSubmit={handleSubmit} className="bg-dark-card p-4 rounded-xl border border-dark-border grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
        <div>
          <label className="block text-sm text-dark-muted mb-1">Start Date</label>
          <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="w-full bg-slate-800 border border-dark-border rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-brand-blue outline-none" />
        </div>
        <div>
          <label className="block text-sm text-dark-muted mb-1">Forecast Days (7-90)</label>
          <input type="number" min="7" max="90" value={days} onChange={(e) => setDays(e.target.value)} className="w-full bg-slate-800 border border-dark-border rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-brand-blue outline-none" />
        </div>
        <div>
          <label className="block text-sm text-dark-muted mb-1">DL Model</label>
          <select value={model} onChange={(e) => setModel(e.target.value)} className="w-full bg-slate-800 border border-dark-border rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-brand-blue outline-none">
            {MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
          </select>
        </div>
        <button type="submit" disabled={loading} className="bg-brand-blue hover:bg-brand-blue/80 text-white px-4 py-2 rounded-lg font-medium transition disabled:opacity-50">
          {loading ? '🔄 Predicting...' : '🚀 Generate Forecast'}
        </button>
      </form>

      {error && <div className="bg-red-500/10 border border-red-500/30 text-red-400 p-3 rounded-lg">{error}</div>}

      {predictions && (
        <>
          <RainfallChart data={predictions} modelName={model} title={`Rainfall Forecast: ${model.toUpperCase()}`} />
          <div className="bg-dark-card rounded-xl border border-dark-border overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="bg-slate-800/50 text-dark-muted uppercase text-xs">
                <tr>
                  <th className="px-4 py-3">Date</th>
                  <th className="px-4 py-3">Predicted (mm)</th>
                  <th className="px-4 py-3">CI Lower (mm)</th>
                  <th className="px-4 py-3">CI Upper (mm)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-dark-border">
                {predictions.map((p, i) => (
                  <tr key={i} className="hover:bg-slate-700/20">
                    <td className="px-4 py-2">{new Date(p.date).toLocaleDateString('en-IN')}</td>
                    <td className="px-4 py-2 font-medium">{p.predicted_rainfall_mm}</td>
                    <td className="px-4 py-2 text-dark-muted">{p.confidence_interval_low}</td>
                    <td className="px-4 py-2 text-dark-muted">{p.confidence_interval_high}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}