'use client';

import React, { useEffect, useState } from 'react';
import { Activity, Clock, Zap } from 'lucide-react';

interface ApiLoadingOverlayProps {
  isVisible: boolean;
  onClose?: () => void;
}

const ApiLoadingOverlay: React.FC<ApiLoadingOverlayProps> = ({ isVisible, onClose }) => {
  const [currentMessage, setCurrentMessage] = useState(0);
  const [secondsElapsed, setSecondsElapsed] = useState(0);

  const messages = [
    { icon: Activity, text: "Initializing AWS Lambda function..." },
    { icon: Zap, text: "Loading PyTorch models..." },
    { icon: Clock, text: "Cold start in progress..." },
    { icon: Activity, text: "Preparing neural networks..." },
  ];

  useEffect(() => {
    if (!isVisible) {
      setSecondsElapsed(0);
      setCurrentMessage(0);
      return;
    }

    // Timer for elapsed seconds
    const secondsTimer = setInterval(() => {
      setSecondsElapsed(prev => prev + 1);
    }, 1000);

    // Message rotation every 3 seconds
    const messageTimer = setInterval(() => {
      setCurrentMessage(prev => (prev + 1) % messages.length);
    }, 3000);

    return () => {
      clearInterval(secondsTimer);
      clearInterval(messageTimer);
    };
  }, [isVisible, messages.length]);

  if (!isVisible) return null;

  const CurrentIcon = messages[currentMessage].icon;
  const progress = Math.min((secondsElapsed / 15) * 100, 100); // AWS Lambda cold start ~10-15s

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white/95 backdrop-blur-md rounded-3xl border border-white/50 shadow-2xl p-8 max-w-md w-full mx-4 text-center">
        {/* Animated Icon */}
        <div className="relative mb-6">
          <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full mx-auto flex items-center justify-center animate-pulse">
            <CurrentIcon className="w-10 h-10 text-white animate-spin" style={{ animationDuration: '2s' }} />
          </div>
          {/* Ripple effect */}
          <div className="absolute inset-0 w-20 h-20 bg-blue-400/30 rounded-full mx-auto animate-ping"></div>
        </div>

        {/* Title */}
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Starting Backend API
        </h2>

        {/* Dynamic Message */}
        <p className="text-lg text-gray-600 mb-6 min-h-[28px] transition-all duration-500 ease-in-out">
          {messages[currentMessage].text}
        </p>

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="flex justify-between text-sm text-gray-500 mb-2">
            <span>Progress</span>
            <span>{secondsElapsed}s elapsed</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-blue-500 to-indigo-600 h-2 rounded-full transition-all duration-1000 ease-out"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>

        {/* Info Cards */}
        <div className="grid grid-cols-2 gap-3 mb-6">
          <div className="bg-blue-50 rounded-xl p-3">
            <div className="text-blue-700 font-semibold text-sm">Expected Time</div>
            <div className="text-blue-900 text-xs">5-15 seconds</div>
          </div>
          <div className="bg-green-50 rounded-xl p-3">
            <div className="text-green-700 font-semibold text-sm">Platform</div>
            <div className="text-green-900 text-xs">AWS Lambda</div>
          </div>
        </div>

        {/* Why this happens */}
        <div className="text-xs text-gray-500 bg-gray-50 rounded-xl p-3">
          <strong>Why the wait?</strong> AWS Lambda experiences a &quot;cold start&quot; when the function hasn&apos;t been used recently.
          Subsequent requests will be much faster!
        </div>

        {/* Skip button after 20 seconds */}
        {secondsElapsed > 20 && onClose && (
          <button
            onClick={onClose}
            className="mt-4 px-4 py-2 text-sm text-gray-600 hover:text-gray-800 transition-colors"
          >
            Continue anyway (API might not be ready)
          </button>
        )}
      </div>
    </div>
  );
};

export default ApiLoadingOverlay;