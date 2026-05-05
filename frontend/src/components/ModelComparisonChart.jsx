import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, BarChart, Bar } from 'recharts';

const MODEL_COLORS = {
  lstm: '#3B82F6',
  bilstm: '#14B8A6',
  gru: '#F59E0B',
  cnn_lstm: '#22C55E',
  transformer: '#8B5CF6',
  autoencoder: '#EF4444',
};

export function MultiLinePredictionChart({ data }) {
  return (
    <div className="bg-dark-card rounded-xl p-4 border border-dark-border">
      <h3 className="text-lg font-semibold mb-4">Model Predictions Comparison</h3>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" className="chart-grid" />
            <XAxis dataKey="date" tick={{ fill: '#94A3B8', fontSize: 11 }} />
            <YAxis tick={{ fill: '#94A3B8' }} />
            <Tooltip contentStyle={{ backgroundColor: '#1E293B', border: '1px solid #334155' }} />
            <Legend />
            {Object.keys(MODEL_COLORS).map(model => (
              <Line 
                key={model} 
                type="monotone" 
                dataKey={model} 
                stroke={MODEL_COLORS[model]} 
                strokeWidth={2} 
                dot={false} 
                name={model.replace('_', ' ')} 
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function RadarMetricsChart({ metrics }) {
  const radarData = Object.entries(metrics || {}).map(([name, val]) => ({
    subject: name.replace('_', ' ').replace('lstm', 'LSTM').replace('bilstm', 'BiLSTM').replace('cnn_lstm', 'CNN-LSTM').replace('autoencoder', 'AutoEncoder'),
    A: Math.max(0, (val.r2 || 0) * 100),
    B: 100 - Math.min(100, (val.mae || 0) * 5),
    C: 100 - Math.min(100, (val.rmse || 0) * 4),
  }));

  return (
    <div className="bg-dark-card rounded-xl p-4 border border-dark-border">
      <h3 className="text-lg font-semibold mb-4">Performance Radar</h3>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
            <PolarGrid className="chart-grid" />
            <PolarAngleAxis dataKey="subject" tick={{ fill: '#94A3B8', fontSize: 11 }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} />
            <Radar name="R² Score" dataKey="A" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.5} />
            <Radar name="MAE (Inverted)" dataKey="B" stroke="#22C55E" fill="#22C55E" fillOpacity={0.3} />
            <Radar name="RMSE (Inverted)" dataKey="C" stroke="#F59E0B" fill="#F59E0B" fillOpacity={0.3} />
            <Legend />
            <Tooltip contentStyle={{ backgroundColor: '#1E293B', border: '1px solid #334155' }} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function R2BarChart({ metrics }) {
  const barData = Object.entries(metrics || {}).map(([name, val]) => ({
    model: name.replace('_', ' '),
    r2: (val.r2 || 0) * 100,
  })).sort((a, b) => b.r2 - a.r2);

  return (
    <div className="bg-dark-card rounded-xl p-4 border border-dark-border">
      <h3 className="text-lg font-semibold mb-4">R² Score Comparison (%)</h3>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={barData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" className="chart-grid" />
            <XAxis type="number" domain={[80, 100]} tick={{ fill: '#94A3B8' }} />
            <YAxis dataKey="model" type="category" tick={{ fill: '#94A3B8' }} width={80} />
            <Tooltip cursor={{ fill: '#1E293B' }} contentStyle={{ backgroundColor: '#1E293B', border: '1px solid #334155' }} />
            <Bar dataKey="r2" name="R² %" fill="#3B82F6" radius={[0, 4, 4, 0]} barSize={20} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}