import { ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function StorageChart({ data }) {
  return (
    <div className="bg-dark-card rounded-xl p-4 border border-dark-border">
      <h3 className="text-lg font-semibold mb-4">Water Storage Projection</h3>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data}>
            <CartesianGrid strokeDasharray="3 3" className="chart-grid" />
            <XAxis dataKey="date" tick={{ fill: '#94A3B8', fontSize: 12 }} tickFormatter={(val) => new Date(val).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' })} />
            <YAxis label={{ value: 'Volume (Liters)', angle: -90, position: 'insideLeft', fill: '#94A3B8' }} tick={{ fill: '#94A3B8', fontSize: 11 }} />
            <Tooltip contentStyle={{ backgroundColor: '#1E293B', border: '1px solid #334155', borderRadius: '8px' }} />
            <Legend />
            <Area type="monotone" dataKey="water_collected_L" stackId="storage" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.4} name="Collected" />
            <Area type="monotone" dataKey="evaporation_loss_L" stackId="loss" stroke="#EF4444" fill="#EF4444" fillOpacity={0.4} name="Evaporation Loss" />
            <Line type="monotone" dataKey="cumulative_storage_L" stroke="#22C55E" strokeWidth={2} dot={false} name="Cumulative Net" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}