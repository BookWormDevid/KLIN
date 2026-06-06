import React from 'react';
import ReactDOM from 'react-dom/client';
import { App } from './pages/App';
import { ServicesProvider, defaultServices } from './core/context/ServicesContext';
import './styles/global.css';
import './index.css';

// Очищаем мок-токен, если он остался после тестов
if (localStorage.getItem('klin_jwt') === 'mock-token') {
    localStorage.removeItem('klin_jwt');
}

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <ServicesProvider services={defaultServices}>
            <App />
        </ServicesProvider>
    </React.StrictMode>
);
