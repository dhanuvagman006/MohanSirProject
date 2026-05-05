import { useState, useEffect } from 'react';
import { predictAllModels, fetchModelMetrics } from '../api/apiClient';
import { MultiLinePredictionChart, RadarMetricsChart, R2BarChart } from '../components/ModelComparisonChart';
import ModelAccuracyCard from '../components/ModelAccuracyCard';

export default function ModelComparison() {
  const [loading, setLoading] = useState(true);
  const [predictions, setPredictions] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [bestModel, setBestModel] = useState('');

  useEffect(() => {
    const loadData = async () => {
      try {
        const [predRes, metRes] = await Promise.all([
          predictAllModels({ start_date: '2024-06-01', days: 30 }),
          fetchModelMetrics()
        ]);
        
        setPredictions(predRes);
        setMetrics(metRes.models || {});
        
        const best = Object.entries(metRes.models || {}).sort((a, b) => (b[1].r2 || 0) - (a[1].r2 || 0))[0];
        setBestModel(best ? best[0] : '');
      } catch (err) {
        console.error('Comparison load failed:', err);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  if (loading) return <div className="flex justify-center p-20"><div className="animate-spin rounded-full h-10 w-10 border-b-2 border-brand-blue"></div></div>;

  const comparisonTable = Object.entries(metrics).map(([name, val]) => ({
    name,
    r2: val.r2 || 0,
    mae: val.mae || 0,
    rmse: val.rmse || 0,
    mape: val.mape || 0,
    params: val.trainable_params?.toLocaleString() || 'N/A',
  }));

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Model Comparison Dashboard</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {comparisonTable.map(row => (
          <ModelAccuracyCard key={row.name} name={row.name} r2={row.r2} mae={row.mae} rmse={row.rmse} isBest={row.name === bestModel} />
        ))}
      </div>

      <MultiLinePredictionChart data={predictions} />
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RadarMetricsChart metrics={metrics} />
        <R2BarChart metrics={metrics} />
      </div>

      <div className="bg-dark-card rounded-xl border border-dark-border overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="bg-slate-800/50 text-dark-muted uppercase text-xs">
            <tr>
              <th className="px-4 py-3">Model</th>
              <th className="px-4 py-3">R²</th>
              <th className="px-4 py-3">MAE (mm)</th>
              <th className="px-4 py-3">RMSE (mm)</th>
              <th className="px-4 py-3">MAPE (%)</th>
              <th className="px-4 py-3">Parameters</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-dark-border">
            {comparisonTable.map(row => (
              <tr key={row.name} className={`hover:bg-slate-700/20 ${row.name === bestModel ? 'bg-yellow-400/5' : ''}`}>
                <td className="px-4 py-3 font-medium capitalize">{row.name.replace('_', ' ')}</td>
                <td className="px-4 py-3 font-bold text-brand-green">{row.r2.toFixed(4)}</td>
                <td className="px-4 py-3">{row.mae.toFixed(2)}</td>
                <td className="px-4 py-3">{row.rmse.toFixed(2)}</td>
                <td className="px-4 py-3">{row.mape.toFixed(2)}</td>
                <td className="px-4 py-3">{row.params}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}