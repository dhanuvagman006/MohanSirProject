export default function ModelAccuracyCard({ name, r2, mae, rmse, isBest = false }) {
  const colors = {
    lstm: 'border-brand-blue',
    bilstm: 'border-brand-teal',
    gru: 'border-brand-amber',
    cnn_lstm: 'border-brand-green',
    transformer: 'border-brand-purple',
    autoencoder: 'border-brand-coral',
  };

  return (
    <div className={`bg-dark-card rounded-xl p-4 border ${isBest ? 'border-2 border-yellow-400 shadow-lg shadow-yellow-400/10' : colors[name] || 'border-dark-border'} relative`}>
      {isBest && <span className="absolute top-2 right-2 bg-yellow-400 text-xs font-bold px-2 py-0.5 rounded text-dark-bg">⭐ BEST</span>}
      <h3 className="font-semibold capitalize mb-2">{name.replace('_', ' ')}</h3>
      <div className="grid grid-cols-3 gap-2 text-sm">
        <div>
          <p className="text-dark-muted">R² Score</p>
          <p className="font-bold text-brand-green">{r2.toFixed(4)}</p>
        </div>
        <div>
          <p className="text-dark-muted">MAE</p>
          <p className="font-bold">{mae.toFixed(2)} mm</p>
        </div>
        <div>
          <p className="text-dark-muted">RMSE</p>
          <p className="font-bold">{rmse.toFixed(2)} mm</p>
        </div>
      </div>
    </div>
  );
}