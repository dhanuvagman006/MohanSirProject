import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchDatasetStats, fetchMonthlyRainfall, fetchAnnualRainfall, fetchModelMetrics } from '../api/apiClient';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';
import { Droplets, Database, TrendingUp, Calendar } from 'lucide-react';

export default function Dashboard() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState(null);
  const [monthly, setMonthly] = useState([]);
  const [annual, setAnnual] = useState([]);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [s, m, a, met] = await Promise.allSettled([
          fetchDatasetStats(),
          fetchMonthlyRainfall(),
          fetchAnnualRainfall(),
          fetchModelMetrics()
        ]);
        if (s.status === 'fulfilled') setStats(s.value);
        if (m.status === 'fulfilled') setMonthly(Object.entries(m.value || {}).map(([name, val]) => ({ name, ...val })));
        if (a.status === 'fulfilled') setAnnual(Object.entries(a.value?.yearly_totals || {}).map(([year, total]) => ({ year, total })));
        if (met.status === 'fulfilled') setMetrics(met.value);
      } catch (err) {
        console.error('Dashboard load failed:', err);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, []);

  const bestR2 = metrics?.models ? Math.max(...Object.values(metrics.models).map(m => m.r2 || 0)) : 0;
  const bestModel = metrics?.best_model || 'N/A';

  if (loading) return <div className="flex justify-center p-20"><div className="animate-spin rounded-full h-10 w-10 border-b-2 border-brand-blue"></div></div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">DK-AquaPredict Dashboard</h1>
        <span className="text-sm text-dark-muted">Dakshina Kannada Region • 2000-2024</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { icon: <Database size={20} />, label: 'Dataset Span', value: `${stats?.total_records?.toLocaleString() || '9,131'} days`, color: 'bg-brand-blue/10 text-brand-blue' },
          { icon: <Droplets size={20} />, label: 'Avg Annual Rainfall', value: `${Math.round(stats?.rainfall.annual_mean || 4000)} mm`, color: 'bg-brand-teal/10 text-brand-teal' },
          { icon: <TrendingUp size={20} />, label: 'DL Models Trained', value: '6 Models', color: 'bg-brand-purple/10 text-brand-purple' },
          { icon: <Calendar size={20} />, label: 'Best Model R²', value: `${bestR2.toFixed(3)} (${bestModel})`, color: 'bg-brand-green/10 text-brand-green' },
        ].map((card, i) => (
          <div key={i} className="bg-dark-card p-4 rounded-xl border border-dark-border flex items-center gap-4">
            <div className={`p-3 rounded-lg ${card.color}`}>{card.icon}</div>
            <div>
              <p className="text-sm text-dark-muted">{card.label}</p>
              <p className="text-xl font-bold">{card.value}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-dark-card rounded-xl p-4 border border-dark-border">
          <h3 className="text-lg font-semibold mb-4">Monthly Average Rainfall (Historical)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={monthly}>
                <CartesianGrid strokeDasharray="3 3" className="chart-grid" />
                <XAxis dataKey="name" tick={{ fill: '#94A3B8' }} />
                <YAxis tick={{ fill: '#94A3B8' }} />
                <Tooltip contentStyle={{ backgroundColor: '#1E293B', border: '1px solid #334155' }} />
                <Bar dataKey="mean" fill="#3B82F6" name="Avg Rainfall (mm)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-dark-card rounded-xl p-4 border border-dark-border">
          <h3 className="text-lg font-semibold mb-4">Annual Rainfall Trend (2000-2024)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={annual}>
                <CartesianGrid strokeDasharray="3 3" className="chart-grid" />
                <XAxis dataKey="year" tick={{ fill: '#94A3B8' }} />
                <YAxis tick={{ fill: '#94A3B8' }} />
                <Tooltip contentStyle={{ backgroundColor: '#1E293B', border: '1px solid #334155' }} />
                <Line type="monotone" dataKey="total" stroke="#14B8A6" strokeWidth={2} dot={false} name="Total (mm)" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap gap-3">
        {[
          { to: '/predict', label: '🌧️ Predict Rainfall', color: 'bg-brand-blue hover:bg-brand-blue/80' },
          { to: '/storage', label: '💧 Water Storage', color: 'bg-brand-teal hover:bg-brand-teal/80' },
          { to: '/irrigation', label: '🌱 Irrigation Schedule', color: 'bg-brand-green hover:bg-brand-green/80' },
          { to: '/compare', label: '📊 Compare Models', color: 'bg-brand-purple hover:bg-brand-purple/80' },
        ].map((btn) => (
          <button key={btn.to} onClick={() => navigate(btn.to)} className={`${btn.color} text-white px-4 py-2 rounded-lg font-medium transition`}>
            {btn.label}
          </button>
        ))}
      </div>
    </div>
  );
}