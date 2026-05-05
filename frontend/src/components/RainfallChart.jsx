import { ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const COLORS = {
  lstm: '#3B82F6',
  bilstm: '#14B8A6',
  gru: '#F59E0B',
  cnn_lstm: '#22C55E',
  transformer: '#8B5CF6',
  autoencoder: '#EF4444',
};

export default function RainfallChart({ data, modelName = 'lstm', title = "Predicted Rainfall" }) {
  const color = COLORS[modelName] || COLORS.lstm;

  return (
    <div className="bg-dark-card rounded-xl p-4 border border-dark-border">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data}>
            <CartesianGrid strokeDasharray="3 3" className="chart-grid" />
            <XAxis dataKey="date" tick={{ fill: '#94A3B8', fontSize: 12 }} tickFormatter={(val) => new Date(val).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' })} />
            <YAxis label={{ value: 'Rainfall (mm)', angle: -90, position: 'insideLeft', fill: '#94A3B8' }} tick={{ fill: '#94A3B8' }} />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1E293B', border: '1px solid #334155', borderRadius: '8px' }}
              labelStyle={{ color: '#F8FAFC' }}
              formatter={(value, name, props) => {
                if (name === 'Predicted') return [`${value} mm`, 'Predicted Rainfall'];
                return null;
              }}
            />
            <Legend />
            <Area 
              type="monotone" 
              dataKey="confidence_interval_low" 
              stackId="ci" 
              stroke="none" 
              fill={color} 
              fillOpacity={0.15} 
              name="CI Lower" 
            />
            <Area 
              type="monotone" 
              dataKey="confidence_interval_high" 
              stackId="ci" 
              stroke="none" 
              fill={color} 
              fillOpacity={0.15} 
              name="CI Upper" 
            />
            <Line 
              type="monotone" 
              dataKey="predicted_rainfall_mm" 
              stroke={color} 
              strokeWidth={2} 
              dot={false} 
              name="Predicted" 
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}