'use client';

import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { TrendingUp, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { PredictionResponse } from '@/lib/api';

interface PredictionResultProps {
  prediction: PredictionResponse | null;
  isLoading?: boolean;
  error?: string | null;
}

const PredictionResult: React.FC<PredictionResultProps> = ({ prediction, isLoading, error }) => {
  if (isLoading) {
    return (
      <div className="bg-white/95 backdrop-blur-sm rounded-2xl p-4 sm:p-6 lg:p-8 shadow-xl border border-gray-100/50">
        <div className="animate-pulse" role="status" aria-label="Loading prediction">
          <div className="h-6 sm:h-8 bg-gray-200 rounded w-3/4 mb-4"></div>
          <div className="h-48 sm:h-64 bg-gray-200 rounded"></div>
          <span className="sr-only">Making prediction...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white/95 backdrop-blur-sm rounded-2xl p-4 sm:p-6 lg:p-8 shadow-xl border border-red-100/50" role="alert">
        <div className="flex items-center gap-3 text-red-600 mb-4">
          <AlertCircle className="w-5 h-5 sm:w-6 sm:h-6 flex-shrink-0" aria-hidden="true" />
          <h3 className="text-lg sm:text-xl font-semibold leading-tight">Prediction Error</h3>
        </div>
        <p className="text-sm sm:text-base text-gray-600 leading-relaxed">{error}</p>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="bg-white/95 backdrop-blur-sm rounded-2xl p-4 sm:p-6 lg:p-8 shadow-xl border border-gray-100/50">
        <div className="text-center text-gray-500">
          <div className="mb-4">
            <div className="w-12 h-12 sm:w-16 sm:h-16 mx-auto bg-gray-100 rounded-full flex items-center justify-center">
              <TrendingUp className="w-6 h-6 sm:w-8 sm:h-8 text-gray-400" aria-hidden="true" />
            </div>
          </div>
          <h3 className="text-lg sm:text-xl font-semibold mb-2 leading-tight">Ready to Predict</h3>
          <p className="text-sm sm:text-base leading-relaxed">Draw a digit and click &quot;Predict&quot; to see the results</p>
        </div>
      </div>
    );
  }

  // Prepare data for the chart
  const chartData = Array.from({ length: 10 }, (_, i) => {
    const pred = prediction.top_k_predictions.find(p => p.class === i);
    return {
      digit: i.toString(),
      confidence: pred ? pred.probability * 100 : 0,
    };
  });

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.9) return '#10b981'; // green
    if (confidence > 0.7) return '#3b82f6'; // blue
    if (confidence > 0.5) return '#f59e0b'; // yellow
    return '#ef4444'; // red
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence > 0.9) return 'Very High';
    if (confidence > 0.7) return 'High';
    if (confidence > 0.5) return 'Moderate';
    return 'Low';
  };

  return (
    <div className="bg-white/95 backdrop-blur-sm rounded-2xl p-4 sm:p-6 lg:p-8 shadow-xl border border-gray-100/50">
      {/* Header */}
      <div className="mb-4 sm:mb-6">
        <div className="flex items-center justify-between mb-3 sm:mb-4 gap-2">
          <h3 className="text-lg sm:text-xl lg:text-2xl font-bold text-gray-800 leading-tight">Prediction Results</h3>
          <div className={cn(
            'px-2 sm:px-3 py-1 rounded-full text-xs sm:text-sm font-medium flex-shrink-0',
            prediction.confidence > 0.9 ? 'bg-green-100 text-green-700' :
            prediction.confidence > 0.7 ? 'bg-blue-100 text-blue-700' :
            prediction.confidence > 0.5 ? 'bg-yellow-100 text-yellow-700' :
            'bg-red-100 text-red-700'
          )}>
            {getConfidenceLabel(prediction.confidence)}
          </div>
        </div>

        {/* Main Prediction */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-4 sm:p-6 mb-4 sm:mb-6">
          <div className="flex items-center justify-between gap-4">
            <div className="min-w-0">
              <p className="text-xs sm:text-sm text-gray-600 mb-1 leading-tight">Predicted Digit</p>
              <p className="text-3xl sm:text-4xl lg:text-5xl font-bold text-gray-800 leading-none">{prediction.predicted_class}</p>
            </div>
            <div className="text-right flex-shrink-0">
              <p className="text-xs sm:text-sm text-gray-600 mb-1 leading-tight">Confidence</p>
              <p className="text-xl sm:text-2xl lg:text-3xl font-bold leading-none" style={{ color: getConfidenceColor(prediction.confidence) }}>
                {(prediction.confidence * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>

        {/* Model Info */}
        <div className="grid grid-cols-2 gap-3 sm:gap-4 mb-4 sm:mb-6">
          <div className="bg-gray-50 rounded-lg p-3 sm:p-4">
            <p className="text-xs sm:text-sm text-gray-600 mb-1 leading-tight">Model</p>
            <p className="font-semibold text-gray-800 text-sm sm:text-base leading-tight">{prediction.model_type.toUpperCase()}</p>
          </div>
          <div className="bg-gray-50 rounded-lg p-3 sm:p-4">
            <p className="text-xs sm:text-sm text-gray-600 mb-1 leading-tight">Parameters</p>
            <p className="font-semibold text-gray-800 text-sm sm:text-base leading-tight">{prediction.model_parameters.toLocaleString()}</p>
          </div>
        </div>
      </div>

      {/* Confidence Chart */}
      <div>
        <h4 className="text-base sm:text-lg font-semibold text-gray-700 mb-3 sm:mb-4 leading-tight">Confidence Distribution</h4>
        <ResponsiveContainer width="100%" height={200} className="sm:!h-[250px]">
          <BarChart data={chartData} margin={{ top: 20, right: 20, left: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="digit"
              tick={{ fill: '#6b7280', fontSize: 12 }}
              axisLine={{ stroke: '#e5e7eb' }}
            />
            <YAxis
              tick={{ fill: '#6b7280', fontSize: 12 }}
              axisLine={{ stroke: '#e5e7eb' }}
              label={{ value: 'Confidence (%)', angle: -90, position: 'insideLeft', style: { fill: '#6b7280', fontSize: 12 } }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                border: '1px solid #e5e7eb',
                borderRadius: '8px',
                fontSize: '14px',
              }}
              formatter={(value: number) => [`${value.toFixed(2)}%`, 'Confidence']}
              labelFormatter={(label) => `Digit ${label}`}
            />
            <Bar dataKey="confidence" radius={[4, 4, 0, 0]}>
              {chartData.map((_, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={
                    index === prediction.predicted_class
                      ? getConfidenceColor(prediction.confidence)
                      : '#e5e7eb'
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Top Predictions */}
      <div className="mt-6">
        <h4 className="text-lg font-semibold text-gray-700 mb-3">Top Predictions</h4>
        <div className="space-y-2">
          {prediction.top_k_predictions.map((pred, index) => (
            <div key={pred.class} className="flex items-center gap-3">
              <div className={cn(
                'w-10 h-10 rounded-lg flex items-center justify-center font-semibold',
                index === 0 ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
              )}>
                {pred.class}
              </div>
              <div className="flex-1">
                <div className="bg-gray-200 rounded-full h-2 overflow-hidden">
                  <div
                    className={cn(
                      'h-full transition-all duration-500',
                      index === 0 ? 'bg-blue-500' : 'bg-gray-400'
                    )}
                    style={{ width: `${pred.probability * 100}%` }}
                  />
                </div>
              </div>
              <span className="text-sm font-medium text-gray-600 w-12 text-right">
                {(pred.probability * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;