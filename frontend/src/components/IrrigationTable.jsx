export default function IrrigationTable({ data }) {
  return (
    <div className="bg-dark-card rounded-xl border border-dark-border overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="bg-slate-800/50 text-dark-muted uppercase text-xs">
            <tr>
              <th className="px-4 py-3">Date</th>
              <th className="px-4 py-3">Crop</th>
              <th className="px-4 py-3">Rainfall (mm)</th>
              <th className="px-4 py-3">Irrigation Needed (mm)</th>
              <th className="px-4 py-3">Volume (L)</th>
              <th className="px-4 py-3">Action</th>
              <th className="px-4 py-3">Saved (L)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-dark-border">
            {data.map((row, idx) => (
              <tr key={idx} className="hover:bg-slate-700/20">
                <td className="px-4 py-3">{new Date(row.date).toLocaleDateString('en-IN')}</td>
                <td className="px-4 py-3 capitalize">{row.crop}</td>
                <td className="px-4 py-3">{row.rainfall_mm}</td>
                <td className="px-4 py-3">{row.irrigation_needed_mm}</td>
                <td className="px-4 py-3">{row.irrigation_volume_L.toLocaleString()}</td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                    row.schedule_action === 'irrigate' ? 'bg-brand-coral/20 text-brand-coral' : 'bg-brand-green/20 text-brand-green'
                  }`}>
                    {row.schedule_action === 'irrigate' ? '💧 Irrigate' : '🌧️ Skip'}
                  </span>
                </td>
                <td className="px-4 py-3 text-brand-green">{row.water_savings_L.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}