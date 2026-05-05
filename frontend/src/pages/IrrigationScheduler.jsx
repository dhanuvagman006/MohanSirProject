import { useState, useEffect } from 'react';
import { usePredictionContext } from '../context/PredictionContext';
import { scheduleIrrigation, fetchSamplePredictions } from '../api/apiClient';
import IrrigationTable from '../components/IrrigationTable';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

const CROPS = ['paddy', 'arecanut', 'coconut', 'banana', 'pepper', 'cashew', 'rubber', 'vegetables'];
const PIE_COLORS = ['#3B82F6', '#22C55E'];

export default function IrrigationScheduler() {
  const { predictions } = usePredictionContext();
  const [crop, setCrop] = useState('paddy');
  const [fieldArea, setFieldArea] = useState(5000);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [summary, setSummary] = useState(null);
  const [chartsData, setChartsData] = useState({ grouped: [], pie: [] });

  useEffect(() => {
    if (predictions && predictions.length > 0) handleSchedule(predictions);
    else fetchSamplePredictions().then(res => handleSchedule(res.predictions || []));
  }, []);

  const handleSchedule = async (preds) => {
    if (!preds?.length) return;
    setLoading(true);
    try {
      const res = await scheduleIrrigation({
        predictions: preds,
        crop,
        field_area_m2: parseFloat(fieldArea)
      });
      setResults(res.daily_schedule);
      setSummary(res.summary);
      
      // Prepare chart data
      setChartsData({
        grouped: preds.map((p, i) => ({
          date: new Date(p.date).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' }),
          Rainfall: p.predicted_rainfall_mm,
          'Crop Req': res.parameters.water_requirement_mm_day,
          'Irrigation Needed': res.daily_schedule[i]?.irrigation_needed_mm || 0
        })),
        pie: [
          { name: 'Skip (Rain Sufficient)', value: res.summary.skip_days },
          { name: 'Irrigate Needed', value: res.summary.irrigation_days }
        ]
      });
    } catch (err) {
      console.error('Irrigation schedule failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Smart Irrigation Scheduler</h1>
      <div className="bg-dark-card p-4 rounded-xl border border-dark-border grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
        <div>
          <label className="block text-sm text-dark-muted mb-1">Crop Type</label>
          <select value={crop} onChange={e => setCrop(e.target.value)} className="w-full bg-slate-800 border border-dark-border rounded-lg px-3 py-2 text-white capitalize">
            {CROPS.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-sm text-dark-muted mb-1">Field Area (m²)</label>
          <input type="number" value={fieldArea} onChange={e => setFieldArea(e.target.value)} className="w-full bg-slate-800 border border-dark-border rounded-lg px-3 py-2 text-white" />
        </div>
        <button onClick={() => predictions ? handleSchedule(predictions) : fetchSamplePredictions().then(r => handleSchedule(r.predictions))} disabled={loading} className="bg-brand-green hover:bg-brand-green/80 text-white px-4 py-2 rounded-lg font-medium transition disabled:opacity-50">
          {loading ? '🔄 Scheduling...' : '🌱 Generate Schedule'}
        </button>
      </div>

      {results && results.length > 0 && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-dark-card rounded-xl p-4 border border-dark-border">
              <h3 className="text-lg font-semibold mb-4">Rainfall vs Crop Requirement</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartsData.grouped}>
                    <CartesianGrid strokeDasharray="3 3" className="chart-grid" />
                    <XAxis dataKey="date" tick={{ fill: '#94A3B8', fontSize: 10 }} />
                    <YAxis tick={{ fill: '#94A3B8' }} />
                    <Tooltip contentStyle={{ backgroundColor: '#1E293B', border: '1px solid #334155' }} />
                    <Legend />
                    <Bar dataKey="Rainfall" fill="#3B82F6" />
                    <Bar dataKey="Crop Req" fill="#F59E0B" />
                    <Bar dataKey="Irrigation Needed" fill="#EF4444" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-dark-card rounded-xl p-4 border border-dark-border">
                <h3 className="text-lg font-semibold mb-2">Schedule Distribution</h3>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={chartsData.pie} cx="50%" cy="50%" outerRadius={60} dataKey="value" label>
                        {chartsData.pie.map((entry, i) => <Cell key={i} fill={PIE_COLORS[i % 2]} />)}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div className="bg-dark-card p-4 rounded-xl border border-dark-border">
                <h3 className="text-lg font-semibold mb-2">Irrigation Summary</h3>
                <div className="space-y-2 text-sm">
                  <p>Irrigation Days: <span className="font-bold text-brand-coral">{summary.irrigation_days}</span></p>
                  <p>Skip Days (Rain OK): <span className="font-bold text-brand-green">{summary.skip_days}</span></p>
                  <p>Total Water Needed: <span className="font-bold text-brand-blue">{summary.total_irrigation_volume_L.toLocaleString()} L</span></p>
                  <p>Total Saved by Rain: <span className="font-bold text-brand-teal">{summary.total_water_saved_L.toLocaleString()} L</span></p>
                </div>
              </div>
            </div>
          </div>
          <IrrigationTable data={results} />
        </>
      )}
    </div>
  );
}