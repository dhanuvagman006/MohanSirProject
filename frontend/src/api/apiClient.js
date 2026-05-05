import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Request interceptor for loading states can be added globally if needed
api.interceptors.request.use((config) => {
  console.log(`🌐 API Call: ${config.method?.toUpperCase()} ${config.url}`);
  return config;
});

api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.detail || error.message || 'Network error';
    console.error('❌ API Error:', message);
    return Promise.reject(new Error(message));
  }
);

export const fetchHealth = () => api.get('/health');
export const fetchDatasetStats = () => api.get('/dataset/stats');
export const fetchModelMetrics = () => api.get('/models/metrics');
export const fetchMonthlyRainfall = () => api.get('/historical/monthly');
export const fetchAnnualRainfall = () => api.get('/historical/annual');
export const fetchSamplePredictions = () => api.get('/sample/predictions');

export const predictRainfall = (data) => api.post('/predict/rainfall', data);
export const predictAllModels = (data) => api.post('/predict/all_models', data);
export const calculateStorage = (data) => api.post('/water/storage', data);
export const scheduleIrrigation = (data) => api.post('/irrigation/schedule', data);

export default api;