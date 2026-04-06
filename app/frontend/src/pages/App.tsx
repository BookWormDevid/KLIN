import React, { useState } from 'react';
import { UploadPage } from './UploadPage';
import { StreamPage } from './StreamPage';

export const App: React.FC = () => {
    const [mode, setMode] = useState<'upload' | 'stream'>('upload');

    return (
        <div className="app-root">
            <header className="app-header">
                <h1>KLIN</h1>
                <div className="mode-switch">
                    <button onClick={() => setMode('upload')} className={mode === 'upload' ? 'active' : ''}>
                        Анализ видео
                    </button>
                    <button onClick={() => setMode('stream')} className={mode === 'stream' ? 'active' : ''}>
                        Прямой эфир
                    </button>
                </div>
            </header>
            {mode === 'upload' ? <UploadPage /> : <StreamPage />}
        </div>
    );
};