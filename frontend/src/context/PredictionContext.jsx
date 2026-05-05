import { createContext, useContext, useState, useCallback } from 'react';

const PredictionContext = createContext();

export const usePredictionContext = () => useContext(PredictionContext);

export const PredictionProvider = ({ children }) => {
  const [predictions, setPredictions] = useState(null);
  const [params, setParams] = useState(null);

  const storePredictions = useCallback((data, config) => {
    setPredictions(data);
    setParams(config);
  }, []);

  const clearPredictions = useCallback(() => {
    setPredictions(null);
    setParams(null);
  }, []);

  return (
    <PredictionContext.Provider value={{ predictions, params, storePredictions, clearPredictions }}>
      {children}
    </PredictionContext.Provider>
  );
};