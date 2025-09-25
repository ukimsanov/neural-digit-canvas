'use client';

import React from 'react';
import { Brain, Zap } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ModelSelectorProps {
  selectedModel: 'linear' | 'cnn';
  onModelChange: (model: 'linear' | 'cnn') => void;
  className?: string;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ selectedModel, onModelChange, className }) => {
  const models = [
    {
      id: 'linear' as const,
      name: 'Linear',
      description: 'Fast & Simple',
      accuracy: '92.4%',
      speed: '~2ms',
      icon: Zap,
      color: 'amber',
    },
    {
      id: 'cnn' as const,
      name: 'CNN',
      description: 'High Accuracy',
      accuracy: '98.2%',
      speed: '~8ms',
      icon: Brain,
      color: 'blue',
    },
  ];

  return (
    <div className={cn('bg-white/95 backdrop-blur-sm rounded-2xl p-4 sm:p-6 shadow-xl border border-gray-100/50', className)}>
      <h3 className="text-lg sm:text-xl font-semibold text-gray-800 mb-4">Select Model</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
        {models.map((model) => {
          const Icon = model.icon;
          const isSelected = selectedModel === model.id;

          return (
            <button
              key={model.id}
              onClick={() => onModelChange(model.id)}
              aria-label={`Select ${model.name} model with ${model.accuracy} accuracy`}
              aria-pressed={isSelected}
              className={cn(
                'relative rounded-xl border-2 p-4 text-left transition-all duration-200 min-h-[112px] touch-manipulation',
                'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500',
                isSelected
                  ? model.color === 'amber' 
                    ? 'border-amber-500 bg-amber-50' 
                    : 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-md'
              )}
            >
              {isSelected && (
                <div className={cn(
                  'absolute top-2 right-2 w-2 h-2 rounded-full',
                  model.color === 'amber' ? 'bg-amber-500' : 'bg-blue-500'
                )} />
              )}

              <div className="flex items-start gap-3">
                <div className={cn(
                  'p-2 rounded-lg flex-shrink-0',
                  isSelected 
                    ? model.color === 'amber' ? 'bg-amber-100' : 'bg-blue-100'
                    : 'bg-gray-100'
                )}>
                  <Icon className={cn(
                    'w-5 h-5',
                    isSelected 
                      ? model.color === 'amber' ? 'text-amber-600' : 'text-blue-600'
                      : 'text-gray-600'
                  )} />
                </div>

                <div className="flex-1 min-w-0">
                  <h4 className="font-semibold text-gray-800 text-sm sm:text-base leading-tight">{model.name}</h4>
                  <p className="text-xs sm:text-sm text-gray-600 mb-2 leading-relaxed">{model.description}</p>

                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-gray-500 block">Accuracy</span>
                      <span className="font-medium text-gray-700 text-sm">{model.accuracy}</span>
                    </div>
                    <div>
                      <span className="text-gray-500 block">Speed</span>
                      <span className="font-medium text-gray-700 text-sm">{model.speed}</span>
                    </div>
                  </div>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default ModelSelector;