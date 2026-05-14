import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from '../src/pages/App';
import { ServicesProvider } from '../src/core/context/ServicesContext';
import { ApiVideoRepository } from './mocks/apiVideoRepository';
import { ApiStreamRepository } from './mocks/apiStreamRepository';
import { ApiFeedbackRepository } from './mocks/apiFeedbackRepository';
import { LocalStorageHistoryRepository } from '../src/core/infrastructure/repositories/localStorageHistoryRepository';
import '../src/styles/global.css';

localStorage.setItem('klin_jwt', 'mock-token');

const mockServices = {
    videoRepo: new ApiVideoRepository() as any,
    streamRepo: new ApiStreamRepository() as any,
    feedbackRepo: new ApiFeedbackRepository() as any,
    historyRepo: new LocalStorageHistoryRepository(),
};

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <ServicesProvider services={mockServices}>
            <App />
        </ServicesProvider>
    </React.StrictMode>
);
