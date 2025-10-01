'use client';

import { useState, useRef, useEffect } from 'react';
import DrawingCanvas, { DrawingCanvasRef } from '@/components/DrawingCanvas';
import PredictionResult from '@/components/PredictionResult';
import ModelSelector from '@/components/ModelSelector';
import ApiLoadingOverlay from '@/components/ApiLoadingOverlay';
import { mnistAPI, PredictionResponse } from '@/lib/api';
import { canvasToBlob, preprocessCanvas } from '@/lib/utils';
import { Eraser, Sparkles, Github, Activity } from 'lucide-react';

export default function Home() {
  const canvasRef = useRef<DrawingCanvasRef>(null);
  const [selectedModel, setSelectedModel] = useState<'linear' | 'cnn'>('cnn');
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [showLoadingOverlay, setShowLoadingOverlay] = useState(true);

  // Check API status with cold start handling
  useEffect(() => {
    let retryCount = 0;
    const maxRetries = 30; // Try for up to 60 seconds (30 * 2s)
    
    const checkApiWithRetry = async () => {
      try {
        await mnistAPI.healthCheck();
        setApiStatus('online');
        // Small delay to show the overlay briefly even on fast connections
        setTimeout(() => setShowLoadingOverlay(false), 500);
      } catch (error) {
        retryCount++;
        if (retryCount >= maxRetries) {
          setApiStatus('offline');
          setShowLoadingOverlay(false);
        } else {
          // Retry every 2 seconds
          setTimeout(checkApiWithRetry, 2000);
        }
      }
    };

    // Start checking after a brief delay for smooth animations
    const timer = setTimeout(checkApiWithRetry, 1000);

    return () => clearTimeout(timer);
  }, []);

  const handlePredict = async () => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current.getCanvas();
    if (!canvas || canvasRef.current.isEmpty()) {
      setError('Please draw a digit first');
      return;
    }

    setIsLoading(true);
    setError(null);
    
    // If API is not online, show appropriate message
    if (apiStatus !== 'online') {
      setError('API is not available. Please wait for the backend to start.');
      setIsLoading(false);
      return;
    }

    try {
      // Preprocess the canvas to 28x28
      const processedCanvas = preprocessCanvas(canvas);
      const blob = await canvasToBlob(processedCanvas);

      // Make prediction
      const result = await mnistAPI.predictImage(blob, selectedModel, 5);
      setPrediction(result);
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err instanceof Error ? err.message : 'Failed to make prediction');
      setApiStatus('offline');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    canvasRef.current?.clear();
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Loading Overlay for API Cold Start */}
      <ApiLoadingOverlay isVisible={showLoadingOverlay} />
      
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-50 hero-header">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14 sm:h-16">
            <div className="flex items-center gap-2 sm:gap-3 min-w-0">
              <div className="p-1.5 sm:p-2 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex-shrink-0">
                <Sparkles className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
              </div>
              <div className="min-w-0">
                <h1 className="text-lg sm:text-xl font-bold text-gray-800 leading-tight truncate">MNIST Classifier</h1>
                <p className="text-xs text-gray-600 leading-tight hidden sm:block">Neural Network Digit Recognition</p>
              </div>
            </div>

            <div className="flex items-center gap-2 sm:gap-4">
              {/* API Status Indicator */}
              <div className="flex items-center gap-1.5 sm:gap-2">
                <Activity className={`w-4 h-4 transition-colors duration-300 ${
                  apiStatus === 'online' ? 'text-green-500' :
                  apiStatus === 'offline' ? 'text-red-500' :
                  'text-gray-400 animate-pulse'
                }`} aria-hidden="true" />
                <div className="text-xs sm:text-sm text-gray-600 min-w-[65px] sm:min-w-[85px] text-left">
                  <span className="hidden sm:inline">API </span>
                  <span className="transition-all duration-300 ease-out">
                    {apiStatus === 'online' ? 'Online' : apiStatus === 'offline' ? 'Offline' : 'Checking...'}
                  </span>
                </div>
              </div>

              <a
                href="https://github.com/ukimsanov/mnist-linear-classifier"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="View source code on GitHub"
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 touch-manipulation"
              >
                <Github className="w-5 h-5 text-gray-700" />
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/5 via-indigo-600/5 to-purple-600/5"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 sm:pt-20 lg:pt-24 pb-16 sm:pb-20 lg:pb-24">
          <div className="text-center mb-16 sm:mb-20">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-md rounded-full border border-blue-200/50 text-sm font-medium text-blue-700 mb-6 sm:mb-8 hero-badge hover:scale-105 hover:shadow-lg transition-transform">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              Neural Network Demo • Try It Live
            </div>
            
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 mb-6 sm:mb-8 leading-tight tracking-tight hero-title">
              Draw a Digit,
              <br />
              <span className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent animate-gradient bg-[length:200%_auto]">
                Watch AI Predict
              </span>
            </h1>
            
            <p className="text-lg sm:text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed mb-8 sm:mb-12 hero-description">
              Experience machine learning in real-time. Choose between our CNN model (98.2% accuracy) 
              or Linear classifier (92.4% accuracy), draw any digit, and see instant predictions.
            </p>
          </div>

          {/* Integrated Demo Section */}
          <div className="relative hero-demo">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 via-transparent to-purple-600/10 rounded-3xl blur-3xl transform -rotate-1 animate-pulse"></div>
            <div className="relative bg-white/90 backdrop-blur-md rounded-3xl border border-white/50 shadow-2xl p-6 sm:p-8 hover:shadow-3xl transition-shadow duration-500 ease-out">
              <div className="grid lg:grid-cols-2 gap-8">
                {/* Left Column - Interactive Demo */}
                <div className="space-y-6">
                  <ModelSelector
                    selectedModel={selectedModel}
                    onModelChange={setSelectedModel}
                  />
                  <div className="bg-gradient-to-br from-gray-50 to-white rounded-2xl p-6 border border-gray-100">
                    <h3 className="text-xl font-semibold text-gray-800 mb-4 text-center">
                      Draw Your Digit
                    </h3>
                    <div className="mb-8">
                      <DrawingCanvas
                        ref={canvasRef}
                        onDrawingChange={() => {
                          setPrediction(null);
                          setError(null);
                        }}
                      />
                    </div>
                    <div className="flex flex-col sm:flex-row gap-3 justify-center">
                      <button
                        onClick={handleClear}
                        aria-label="Clear the drawing canvas"
                        className="flex items-center justify-center gap-2 px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold rounded-xl transition-all duration-300 hover:scale-105 hover:shadow-lg active:scale-95 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-300 min-h-[48px] group"
                      >
                        <Eraser className="w-5 h-5 group-hover:rotate-12 transition-transform duration-200" />
                        Clear
                      </button>
                      <button
                        onClick={handlePredict}
                        disabled={isLoading || apiStatus === 'offline'}
                        aria-label={`Predict digit using ${selectedModel} model`}
                        className="flex items-center justify-center gap-2 px-8 py-3 bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white font-semibold rounded-xl transition-all duration-300 hover:scale-105 hover:shadow-xl active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 min-h-[48px] group relative overflow-hidden"
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-indigo-700 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                        <Sparkles className={`w-5 h-5 relative z-10 transition-transform duration-200 ${isLoading ? 'animate-bounce' : 'group-hover:scale-110 group-hover:rotate-12'}`} />
                        <span className="relative z-10">{isLoading ? 'Predicting...' : 'Predict'}</span>
                      </button>
                    </div>
                  </div>
                </div>

                {/* Right Column - Results */}
                <div>
                  <PredictionResult
                    prediction={prediction}
                    isLoading={isLoading}
                    error={error}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* API Offline Warning */}
          {apiStatus === 'offline' && (
            <div className="mb-4 sm:mb-6 p-4 bg-red-50 border border-red-200 rounded-xl" role="alert" aria-live="polite">
              <p className="text-red-700 text-sm sm:text-base">
                ⚠️ API is offline. Please wait a little bit more for the FastAPI backend server to start
              </p>
            </div>
          )}

          {/* Info Section */}
          <div className="mt-16 sm:mt-20 lg:mt-24 mb-8 sm:mb-12 animate-in fade-in slide-in-from-bottom duration-1000 ease-out delay-1500">
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
          <div className="bg-white rounded-xl p-4 sm:p-6 shadow-md border border-gray-100 hover:shadow-xl hover:scale-105 transition-all duration-300 ease-out cursor-pointer group">
            <h3 className="font-semibold text-gray-800 mb-2 text-sm sm:text-base leading-tight group-hover:text-blue-600 transition-colors duration-200">🎯 High Accuracy</h3>
            <p className="text-xs sm:text-sm text-gray-600 leading-relaxed group-hover:text-gray-700 transition-colors duration-200">
              Our CNN model achieves 98.2% accuracy on the MNIST test dataset,
              trained on 60,000 handwritten digits.
            </p>
          </div>
          <div className="bg-white rounded-xl p-4 sm:p-6 shadow-md border border-gray-100 hover:shadow-xl hover:scale-105 transition-all duration-300 ease-out cursor-pointer group">
            <h3 className="font-semibold text-gray-800 mb-2 text-sm sm:text-base leading-tight group-hover:text-indigo-600 transition-colors duration-200">⚡ Real-time Inference</h3>
            <p className="text-xs sm:text-sm text-gray-600 leading-relaxed group-hover:text-gray-700 transition-colors duration-200">
              Get instant predictions with inference times under 10ms,
              perfect for interactive applications.
            </p>
          </div>
          <div className="bg-white rounded-xl p-4 sm:p-6 shadow-md border border-gray-100 sm:col-span-2 lg:col-span-1 hover:shadow-xl hover:scale-105 transition-all duration-300 ease-out cursor-pointer group">
            <h3 className="font-semibold text-gray-800 mb-2 text-sm sm:text-base leading-tight group-hover:text-purple-600 transition-colors duration-200">🤖 Two Models</h3>
            <p className="text-xs sm:text-sm text-gray-600 leading-relaxed group-hover:text-gray-700 transition-colors duration-200">
              Choose between a fast linear classifier (92.4%) or a more accurate CNN (98.2%),
              each optimized for different use cases.
            </p>
            </div>
          </div>
        </div>

        </div>

        {/* Decorative elements */}
        <div className="absolute top-20 left-10 w-20 h-20 bg-gradient-to-br from-blue-400/20 to-indigo-400/20 rounded-full blur-xl animate-float"></div>
        <div className="absolute bottom-20 right-10 w-32 h-32 bg-gradient-to-br from-purple-400/20 to-pink-400/20 rounded-full blur-xl animate-float-delayed"></div>
      </section>

      {/* Footer */}
      <footer className="mt-12 sm:mt-16 bg-white/80 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
          <div className="text-center text-gray-600">
            <p className="mb-2 text-sm sm:text-base leading-relaxed">
              Built with Next.js, TypeScript, and Tailwind CSS v4
            </p>
            <p className="text-xs sm:text-sm leading-relaxed">
              © 2025 MNIST Classifier |
              <a
                href="https://github.com/ukimsanov/mnist-linear-classifier"
                className="ml-1 text-blue-600 hover:text-blue-700 focus:outline-none focus:underline"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="View source code on GitHub"
              >
                View Source
              </a>
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
