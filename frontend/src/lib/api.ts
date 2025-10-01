import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface PredictionResponse {
  predicted_class: number;
  confidence: number;
  top_k_predictions: Array<{
    class: number;
    probability: number;
  }>;
  model_type: string;
  model_parameters: number;
}

export interface ModelInfo {
  name: string;
  type: string;
  parameters: number;
  loaded: boolean;
}

export interface HealthResponse {
  status: string;
  models_loaded: string[];
  version: string;
}

export const mnistAPI = {
  async healthCheck(): Promise<HealthResponse> {
    const response = await api.get<HealthResponse>('/');
    return response.data;
  },

  async predictImage(
    imageFile: File | Blob,
    modelType: 'linear' | 'cnn' = 'cnn',
    topK: number = 3
  ): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await api.post<PredictionResponse>(
      `/predict?model_type=${modelType}&top_k=${topK}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  },

  async predictBase64(
    base64Image: string,
    modelType: 'linear' | 'cnn' = 'cnn',
    topK: number = 3
  ): Promise<PredictionResponse> {
    const response = await api.post<PredictionResponse>('/predict/base64', {
      image: base64Image,
      model_type: modelType,
      top_k: topK,
    });
    return response.data;
  },

  async getModels(): Promise<{ models: ModelInfo[] }> {
    const response = await api.get<{ models: ModelInfo[] }>('/models');
    return response.data;
  },

  async getModelInfo(modelType: string) {
    const response = await api.get(`/model/${modelType}/info`);
    return response.data;
  },
};